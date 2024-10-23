# An official reimplemented version of Marigold training script.
# Last modified: 2024-04-29
#
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# If you use or adapt this code, please attribute to https://github.com/prs-eth/marigold.
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
import os
import shutil
import cv2
from datetime import datetime
from typing import List, Union

import numpy as np
import torch
from diffusers import DDPMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from marigold.marigold_pipeline import MarigoldPipeline, MarigoldOutput
from src.util import metric
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dic_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.alignment import align_depth_least_square
from src.util.seeding import generate_seed_sequence


from perspective2d import PerspectiveFields


class MarigoldTrainer:
    def __init__(
        self,
        cfg: OmegaConf,
        model: MarigoldPipeline,
        train_dataloader: DataLoader,
        device,
        base_ckpt_dir,
        out_dir_ckpt,
        out_dir_eval,
        out_dir_vis,
        accumulation_steps: int,
        val_dataloaders: List[DataLoader] = None,
        vis_dataloaders: List[DataLoader] = None,
    ):
        self.cfg: OmegaConf = cfg
        self.model: MarigoldPipeline = model
        self.device = device
        self.seed: Union[int, None] = (
            self.cfg.trainer.init_seed
        )  # used to generate seed sequence, set to `None` to train w/o seeding
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader: DataLoader = train_dataloader
        self.val_loaders: List[DataLoader] = val_dataloaders
        self.vis_loaders: List[DataLoader] = vis_dataloaders
        self.accumulation_steps: int = accumulation_steps

        # Adapt input layers
        if 8 != self.model.unet.config["in_channels"]:
            self._replace_unet_conv_in_zero_intialization()

        # Encode empty text prompt
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)

        self.model.unet.enable_xformers_memory_efficient_attention()

        # Trainability
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.requires_grad_(True)

        # Optimizer !should be defined after input layer is adapted
        lr = self.cfg.lr
        self.optimizer = Adam(self.model.unet.parameters(), lr=lr)

        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)

        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)
        self.pf_loss = torch.nn.MSELoss()
        self.pf_loss.requires_grad = True


        # Training noise scheduler
        self.training_noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            os.path.join(
                base_ckpt_dir,
                cfg.trainer.training_noise_scheduler.pretrained_path,
                "scheduler",
            )
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        assert (
            self.prediction_type == self.model.scheduler.config.prediction_type
        ), "Different prediction types"
        self.scheduler_timesteps = (
            self.training_noise_scheduler.config.num_train_timesteps
        )

        # Eval metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        # main metric for best checkpoint saving
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        assert (
            self.main_val_metric in cfg.eval.eval_metrics
        ), f"Main eval metric `{self.main_val_metric}` not found in evaluation metrics."
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8

        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.gt_depth_type = self.cfg.gt_depth_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period

        # Multi-resolution noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = (
                self.cfg.multi_res_noise.downscale_strategy
            )

        # Internal variables
        self.epoch = 1
        self.n_batch_in_epoch = 0  # batch index in the epoch, used when resume training
        self.effective_iter = 0  # how many times optimizer.step() is called
        self.in_evaluation = False
        self.global_seed_sequence: List = []  # consistent global seed sequence, used to seed random generator, to ensure consistency when resuming


        # Perspective Field extractor
        version = 'Paramnet-360Cities-edina-centered'
        self.pf_model = PerspectiveFields(version).eval().cuda()


    def _replace_unet_conv_in(self):
        # replace the first layer to accept 8 in_channels
        _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.model.unet.conv_in.bias.clone()  # [320]
        _weight = _weight.repeat((1, 2, 1, 1))  # Keep selected channel(s)
        # half the activation magnitude
        _weight *= 0.5
        # new conv_in channel
        _n_convin_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        self.model.unet.config["in_channels"] = 8
        logging.info("Unet config is updated")
        return


    def _replace_unet_conv_in_zero_intialization(self):
        # replace the first layer to accept 8 in_channels
        _weight = self.model.unet.conv_in.weight.clone()  # [320, 4, 3, 3]
        _bias = self.model.unet.conv_in.bias.clone()  # [320]
        _weight_add = torch.zeros(_weight.shape)
        _weight = torch.cat((_weight_add, _weight), 1) # [320, 8, 3, 3]
        # new conv_in channel
        _n_convin_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        logging.info("Unet conv_in layer is replaced")
        # replace config
        self.model.unet.config["in_channels"] = 8
        logging.info("Unet config is updated with zero initialization")
        return

    
    def train(self, t_end=None):
        logging.info("Start training")

        device = self.device
        self.model.to(device)

        if self.in_evaluation:
            logging.info(
                "Last evaluation was not finished, will do evaluation before continue training."
            )
            self.validate()

        self.train_metrics.reset()
        accumulated_step = 0

        # self.visualize_contrastive()
        # return

        for epoch in range(self.epoch, self.max_epoch + 1):
            self.epoch = epoch
            logging.debug(f"epoch: {self.epoch}")

            # Skip previous batches when resume
            for batch in skip_first_batches(self.train_loader, self.n_batch_in_epoch):
                self.model.unet.train()

                # globally consistent random generators
                if self.seed is not None:
                    local_seed = self._get_next_seed()
                    rand_num_generator = torch.Generator(device=device)
                    rand_num_generator.manual_seed(local_seed)
                else:
                    rand_num_generator = None

                # >>> With gradient accumulation >>>

                # We just need the perspective field
                field = batch["field"].to(device).to(torch.float32)
                # best_pf = batch["best_pf"].to(device).to(torch.float32)
                worst_pf = batch["worst_pf"].to(device).to(torch.float32)


                batch_size = field.shape[0]
                with torch.no_grad():
                    # Encode field depth
                    field_latent = self.model.encode_field(field)  # [B, 4, h, w]

                num_inference_steps = 15

                self.model.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.model.scheduler.timesteps

                # Sample noise
                rgb_latent = torch.randn(
                    field_latent.shape,
                    device=device,
                    generator=rand_num_generator,
                )  # [B, 4, h, w]

                # Text embedding
                text_embed = self.empty_text_embed.to(device).repeat(
                    (batch_size, 1, 1)
                )  # [B, 77, 1024]

                guidance_scale = 7.5

                # guidance embedding
                uncond_input = self.model.tokenizer(
                    [""] * rgb_latent.shape[0], padding="max_length", max_length=self.model.tokenizer.model_max_length, return_tensors="pt"
                )
                with torch.no_grad():
                    uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(device))[0]  
                text_embeddings = torch.cat([uncond_embeddings, text_embed])

                rgb_latent = rgb_latent * self.model.scheduler.init_noise_sigma


                for _, t in enumerate(timesteps):            
                    unet_input = torch.cat(
                        [field_latent, rgb_latent], dim=1
                    )  # this order is important

                    # guidance
                    unet_input = torch.cat([unet_input] * 2)
                    unet_input =  self.model.scheduler.scale_model_input(unet_input, t)

                    # predict the noise residual
                    noise_pred = self.model.unet(
                        unet_input, t, encoder_hidden_states=text_embeddings
                    ).sample  # [B, 4, h, w]

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    rgb_latent = self.model.scheduler.step(
                        noise_pred, t, rgb_latent
                    ).prev_sample

                rgb = self.model.decode_rgb(rgb_latent)

                rgb = torch.clip(rgb, -1.0, 1.0)
                rgb = ((rgb + 1.0) / 2.0).squeeze() * 255

                inputs = {"image": rgb, "height": rgb.shape[1], "width": rgb.shape[2]}
                generated_field = self.pf_model.forward([inputs])[0]

                latitude_map = generated_field['pred_latitude_original']
                gravity_maps = generated_field['pred_gravity_original']
                latitude_map = latitude_map / 90
                joined_maps = torch.cat([gravity_maps, latitude_map.unsqueeze(0),], dim = 0)
                joined_maps = joined_maps.unsqueeze(0)


                # compute the loss
                contrastive_loss = self.contrastive_loss(joined_maps, field, worst_pf)
                # print("contrastive loss", contrastive_loss)
                consistency_loss = self.pf_loss(joined_maps, field)
                # print("consistency loss", consistency_loss)
                loss = contrastive_loss + consistency_loss

                self.train_metrics.update("loss", loss.item())

                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                accumulated_step += 1

                self.n_batch_in_epoch += 1
                # Practical batch end

                # Perform optimization step
                if accumulated_step >= self.gradient_accumulation_steps:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    accumulated_step = 0

                    self.effective_iter += 1

                    # Log to tensorboard
                    accumulated_loss = self.train_metrics.result()["loss"]
                    tb_logger.log_dic(
                        {
                            f"train/{k}": v
                            for k, v in self.train_metrics.result().items()
                        },
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "lr",
                        self.lr_scheduler.get_last_lr()[0],
                        global_step=self.effective_iter,
                    )
                    tb_logger.writer.add_scalar(
                        "n_batch_in_epoch",
                        self.n_batch_in_epoch,
                        global_step=self.effective_iter,
                    )
                    logging.info(
                        f"iter {self.effective_iter:5d} (epoch {epoch:2d}): loss={accumulated_loss:.5f}"
                    )
                    self.train_metrics.reset()

                    # Per-step callback
                    self._train_step_callback()

                    # End of training
                    if self.max_iter > 0 and self.effective_iter >= self.max_iter:
                        self.save_checkpoint(
                            ckpt_name=self._get_backup_ckpt_name(),
                            save_train_state=False,
                        )
                        logging.info("Training ended.")
                        return
                    # Time's up
                    elif t_end is not None and datetime.now() >= t_end:
                        self.save_checkpoint(ckpt_name="latest", save_train_state=True)
                        logging.info("Time is up, training paused.")
                        return

                    torch.cuda.empty_cache()
                    # <<< Effective batch end <<<

            # Epoch end
            self.n_batch_in_epoch = 0


    def contrastive_loss(self, pf_gen, pf_best, pf_worst):
        best_loss = torch.nn.MSELoss()(pf_gen, pf_best) #torch.norm(pf_gen - pf_best, p=2)
        worst_loss = torch.nn.MSELoss()(pf_gen, pf_worst) #torch.norm(pf_gen - pf_worst, p=2)
        return best_loss - worst_loss

    @staticmethod
    def stack_depth_images(depth_in):
        if 4 == len(depth_in.shape):
            stacked = depth_in.repeat(1, 3, 1, 1)
        elif 3 == len(depth_in.shape):
            stacked = depth_in.unsqueeze(1)
            stacked = depth_in.repeat(1, 3, 1, 1)
        return stacked

    def _train_step_callback(self):
        """Executed after every iteration"""
        # Save backup (with a larger interval, without training states)
        if self.backup_period > 0 and 0 == self.effective_iter % self.backup_period:
            self.save_checkpoint(
                ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
            )

        _is_latest_saved = False
        # Validation
        if self.val_period > 0 and 0 == self.effective_iter % self.val_period:
            self.in_evaluation = True  # flag to do evaluation in resume run if validation is not finished
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)
            _is_latest_saved = True
            #self.validate()
            self.in_evaluation = False
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Save training checkpoint (can be resumed)
        if (
            self.save_period > 0
            and 0 == self.effective_iter % self.save_period
            and not _is_latest_saved
        ):
            self.save_checkpoint(ckpt_name="latest", save_train_state=True)

        # Visualization
        if self.vis_period > 0 and 0 == self.effective_iter % self.vis_period:
            self.visualize()

    '''
    def validate(self):
        for i, val_loader in enumerate(self.val_loaders):
            val_dataset_name = val_loader.dataset.disp_name
            val_metric_dic = self.validate_single_dataset(
                data_loader=val_loader, metric_tracker=self.val_metrics
            )
            logging.info(
                f"Iter {self.effective_iter}. Validation metrics on `{val_dataset_name}`: {val_metric_dic}"
            )
            tb_logger.log_dic(
                {f"val/{val_dataset_name}/{k}": v for k, v in val_metric_dic.items()},
                global_step=self.effective_iter,
            )
            # save to file
            eval_text = eval_dic_to_text(
                val_metrics=val_metric_dic,
                dataset_name=val_dataset_name,
                sample_list_path=val_loader.dataset.filename_ls_path,
            )
            _save_to = os.path.join(
                self.out_dir_eval,
                f"eval-{val_dataset_name}-iter{self.effective_iter:06d}.txt",
            )
            with open(_save_to, "w+") as f:
                f.write(eval_text)

            # Update main eval metric
            if 0 == i:
                main_eval_metric = val_metric_dic[self.main_val_metric]
                if (
                    "minimize" == self.main_val_metric_goal
                    and main_eval_metric < self.best_metric
                    or "maximize" == self.main_val_metric_goal
                    and main_eval_metric > self.best_metric
                ):
                    self.best_metric = main_eval_metric
                    logging.info(
                        f"Best metric: {self.main_val_metric} = {self.best_metric} at iteration {self.effective_iter}"
                    )
                    # Save a checkpoint
                    self.save_checkpoint(
                        ckpt_name=self._get_backup_ckpt_name(), save_train_state=False
                    )

    '''

    def visualize_contrastive(self):
        vis_out_dir = "/share/data/p2p/yz5880/contrastive_samples"
        os.makedirs(vis_out_dir, exist_ok=True)
        _ = self.validate_contrastive(
            data_loader=self.train_loader,
            metric_tracker=self.val_metrics,
            save_to_dir=vis_out_dir,
        )


    @torch.no_grad()
    def validate_contrastive(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        self.model.to(self.device)
        metric_tracker.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))


        img_dir = os.path.join(save_to_dir, "images")
        field_dir = os.path.join(save_to_dir, "fields")
        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):
            
            assert 1 == data_loader.batch_size
            # Read input field
            # print(batch)
            field_in = batch["field"].to(self.device).to(torch.float32)
            rgb_in = batch["image"].to(self.device).to(torch.float32)
            # [1, 3, H, W]

        
            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # generate 10 image for each perspective field, and generate perspective
            # field for each of them
            for j in range(5):
                # Predict depth
                pipe_out: MarigoldOutput = self.model(
                    field_in,
                    denoising_steps=self.cfg.validation.denoising_steps,
                    ensemble_size=self.cfg.validation.ensemble_size,
                    processing_res=self.cfg.validation.processing_res,
                    match_input_res=self.cfg.validation.match_input_res,
                    generator=generator,
                    batch_size=1,  # use batch size 1 to increase reproducibility
                    color_map=None,
                    show_progress_bar=False,
                    resample_method=self.cfg.validation.resample_method,
                )

                image_pred: Image.Image = pipe_out.image
                field_pred: np.ndarray = pipe_out.field

                if save_to_dir is not None:
                    output_dir_jpg = os.path.join(img_dir, "image_" + str(i))
                    output_dir_field = os.path.join(field_dir, "field_" + str(i))
                    os.makedirs(output_dir_jpg, exist_ok=True)
                    os.makedirs(output_dir_field, exist_ok=True)
                    
                    # save image
                    pred_name_base = str(j) + "_pred"
                    jpg_save_path = os.path.join(output_dir_jpg, f"{pred_name_base}.jpg")
                    if os.path.exists(jpg_save_path):
                        logging.warning(f"Existing file: '{jpg_save_path}' will be overwritten")
                    image_pred.save(jpg_save_path)

                    # Save field
                    img_np = np.asarray(image_pred)
                    img_np = torch.as_tensor(img_np.astype("float32").transpose(2, 0, 1))
                    img_input =  {"image": img_np, "height": img_np.shape[1], "width": img_np.shape[2]}

                    field_map = self.pf_model.forward([img_input])[0]
                    latitude_map = field_map['pred_latitude_original']
                    gravity_maps = field_map['pred_gravity_original']
                    joined_maps = torch.cat([gravity_maps, latitude_map.unsqueeze(0),], dim = 0)

                    field_save_path = os.path.join(output_dir_field, f"{pred_name_base}.pt")
                    if os.path.exists(field_save_path):
                        logging.warning(f"Existing file: '{field_save_path}' will be overwritten")
                    torch.save(joined_maps, field_save_path)

                if j == 0:
                    og_field_save_path = os.path.join(output_dir_field, f"original.pt")
                    if os.path.exists(og_field_save_path):
                        logging.warning(f"Existing file: '{og_field_save_path}' will be overwritten")
                    torch.save(field_pred, og_field_save_path)


        return metric_tracker.result()




    def visualize(self):
        for val_loader in self.vis_loaders:
            vis_dataset_name = val_loader.dataset.disp_name
            vis_out_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name(), vis_dataset_name
            )
            os.makedirs(vis_out_dir, exist_ok=True)
            _ = self.validate_single_dataset(
                data_loader=val_loader,
                metric_tracker=self.val_metrics,
                save_to_dir=vis_out_dir,
            )



    @torch.no_grad()
    def validate_single_dataset(
        self,
        data_loader: DataLoader,
        metric_tracker: MetricTracker,
        save_to_dir: str = None,
    ):
        self.model.to(self.device)
        metric_tracker.reset()

        # Generate seed sequence for consistent evaluation
        val_init_seed = self.cfg.validation.init_seed
        val_seed_ls = generate_seed_sequence(val_init_seed, len(data_loader))

        for i, batch in enumerate(
            tqdm(data_loader, desc=f"evaluating on {data_loader.dataset.disp_name}"),
            start=1,
        ):

            if i >= 10:
                break 
            
            assert 1 == data_loader.batch_size
            # Read input field
            # print(batch)
            field_in = batch["field"].to(self.device).to(torch.float32)
            rgb_in = batch["image"].to(self.device).to(torch.float32)
            # [1, 3, H, W]

            # Random number generator
            seed = val_seed_ls.pop()
            if seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)

            # Predict depth
            pipe_out: MarigoldOutput = self.model(
                field_in,
                denoising_steps=self.cfg.validation.denoising_steps,
                ensemble_size=self.cfg.validation.ensemble_size,
                processing_res=self.cfg.validation.processing_res,
                match_input_res=self.cfg.validation.match_input_res,
                generator=generator,
                batch_size=1,  # use batch size 1 to increase reproducibility
                color_map=None,
                show_progress_bar=False,
                resample_method=self.cfg.validation.resample_method,
            )

            image_pred: Image.Image = pipe_out.image
            field_pred: np.ndarray = pipe_out.field
            vis_pred: Image.Image = pipe_out.field_visualized
            
            if save_to_dir is not None:
                output_dir_jpg = os.path.join(save_to_dir, "image")
                output_dir_field = os.path.join(save_to_dir, "field")
                output_dir_vis = os.path.join(save_to_dir, "field_visualization")
                os.makedirs(output_dir_jpg, exist_ok=True)
                os.makedirs(output_dir_field, exist_ok=True)
                os.makedirs(output_dir_vis, exist_ok=True)

                
                 # save image
                pred_name_base = str(i) + "_pred"
                jpg_save_path = os.path.join(output_dir_jpg, f"{pred_name_base}.jpg")
                if os.path.exists(jpg_save_path):
                    logging.warning(f"Existing file: '{jpg_save_path}' will be overwritten")
                image_pred.save(jpg_save_path)

                # Save field
                
                field_save_path = os.path.join(output_dir_field, f"{pred_name_base}.pt")
                if os.path.exists(field_save_path):
                    logging.warning(f"Existing file: '{field_save_path}' will be overwritten")
                torch.save(field_pred, field_save_path,)

                # save visualized image
                vis_save_path = os.path.join(output_dir_vis, f"{pred_name_base}.jpg")
                if os.path.exists(vis_save_path):
                    logging.warning(f"Existing file: '{vis_save_path}' will be overwritten")
                vis_pred.save(vis_save_path)

        return metric_tracker.result()
        

    def _get_next_seed(self):
        if 0 == len(self.global_seed_sequence):
            self.global_seed_sequence = generate_seed_sequence(
                initial_seed=self.seed,
                length=self.max_iter * self.gradient_accumulation_steps,
            )
            logging.info(
                f"Global seed sequence is generated, length={len(self.global_seed_sequence)}"
            )
        return self.global_seed_sequence.pop()

    def save_checkpoint(self, ckpt_name, save_train_state):
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving checkpoint to: {ckpt_dir}")
        # Backup previous checkpoint
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
            temp_ckpt_dir = os.path.join(
                os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}"
            )
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
            logging.debug(f"Old checkpoint is backed up at: {temp_ckpt_dir}")

        # Save UNet
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=False)
        logging.info(f"UNet is saved to: {unet_path}")

        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            train_state_path = os.path.join(ckpt_dir, "trainer.ckpt")
            torch.save(state, train_state_path)
            # iteration indicator
            f = open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w")
            f.close()

            logging.info(f"Trainer state is saved to: {train_state_path}")

        # Remove temp ckpt
        if temp_ckpt_dir is not None and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logging.debug("Old checkpoint backup is removed.")

    def load_checkpoint(
        self, ckpt_path, load_trainer_state=True, resume_lr_scheduler=True
    ):
        logging.info(f"Loading checkpoint from: {ckpt_path}")
        # Load UNet
        _model_path = os.path.join(ckpt_path, "unet", "diffusion_pytorch_model.bin")
        self.model.unet.load_state_dict(
            torch.load(_model_path, map_location=self.device)
        )
        self.model.unet.to(self.device)
        logging.info(f"UNet parameters are loaded from {_model_path}")

        # Load training states
        if load_trainer_state:
            checkpoint = torch.load(os.path.join(ckpt_path, "trainer.ckpt"))
            self.effective_iter = checkpoint["effective_iter"]
            self.epoch = checkpoint["epoch"]
            self.n_batch_in_epoch = checkpoint["n_batch_in_epoch"]
            self.in_evaluation = checkpoint["in_evaluation"]
            self.global_seed_sequence = checkpoint["global_seed_sequence"]

            self.best_metric = checkpoint["best_metric"]

            self.optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(f"optimizer state is loaded from {ckpt_path}")

            if resume_lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                logging.info(f"LR scheduler state is loaded from {ckpt_path}")

        logging.info(
            f"Checkpoint loaded from: {ckpt_path}. Resume from iteration {self.effective_iter} (epoch {self.epoch})"
        )
        return

    def _get_backup_ckpt_name(self):
        return f"iter_{self.effective_iter:06d}"
