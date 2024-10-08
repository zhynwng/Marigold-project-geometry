# Finetune pipeline to fix projective geometry of diffusion model


import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import PreTrainedModel, PretrainedConfig


from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)

from perspective2d.utils import draw_perspective_fields



class FinetuneOutput(BaseOutput):
    """
    Output class for finetune pipeline.

    Args:
        image: Image generated from the model
        field: perspective field generated from model, corresponding to the 
            image generated
        field_visualized: visualized perspective field on top of the image
    """

    image: Image.Image
    field: np.ndarray
    field_visualized: Image.Image



class FinetunePipeline(DiffusionPipeline):
    """
    Pipeline for finetuning diffusion model to improve projective geometry

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            U-Net that denoise image and field latent
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
        scale_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are scale-invariant. This value must be set in
            the model config. When used together with the `shift_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        shift_invariant (`bool`, *optional*):
            A model property specifying whether the predicted depth maps are shift-invariant. This value must be set in
            the model config. When used together with the `scale_invariant=True` flag, the model is also called
            "affine-invariant". NB: overriding this value is not supported.
        default_denoising_steps (`int`, *optional*):
            The minimum number of denoising diffusion steps that are required to produce a prediction of reasonable
            quality with the given model. This value must be set in the model config. When the pipeline is called
            without explicitly setting `num_inference_steps`, the default value is used. This is required to ensure
            reasonable results with various model flavors compatible with the pipeline, such as those relying on very
            short denoising schedules (`LCMScheduler`) and those with full diffusion schedules (`DDIMScheduler`).
        default_processing_resolution (`int`, *optional*):
            The recommended value of the `processing_resolution` parameter of the pipeline. This value must be set in
            the model config. When the pipeline is called without explicitly setting `processing_resolution`, the
            default value is used. This is required to ensure reasonable results with various model flavors trained
            with varying optimal processing resolution values.
    """

    rgb_latent_scale_factor = 0.18215
    field_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler, LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.register_to_config(
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )


        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.empty_text_embed = None

        self.latent_shape = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        input_prompts: Optional[torch.Tensor] = None,
        denoising_steps: Optional[int] = None,
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        generator: Union[torch.Generator, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> FinetuneOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_prompts : Optional?
            denoising_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            processing_res (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, processes at the original image resolution. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            generator (`torch.Generator`, *optional*, defaults to `None`)
                Random generator for initial noise generation.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            scale_invariant (`str`, *optional*, defaults to `True`):
                Flag of scale-invariant prediction, if True, scale will be adjusted from the raw prediction.
            shift_invariant (`str`, *optional*, defaults to `True`):
                Flag of shift-invariant prediction, if True, shift will be adjusted from the raw prediction, if False, near plane will be fixed at 0m.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """
        # Model-specific optimal default values leading to fast and reasonable results.
        if denoising_steps is None:
            denoising_steps = self.default_denoising_steps
        if processing_res is None:
            processing_res = self.default_processing_resolution

        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            # convert to torch tensor [H, W, rgb] -> [rgb, H, W]
            rgb = pil_to_tensor(input_image)
            rgb = rgb.unsqueeze(0)  # [1, rgb, H, W]
            # print("rgb", rgb.shape)
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        noise = torch.randn(
                rgb.shape,
                device=rgb.device,
                generator=generator,
            )
        cat_rgb = torch.cat([rgb, noise], dim=1)
        # print("cat rgb shape", cat_rgb.shape)


        # ----------------- generating Image and field ----------------
        # generate  image and field maps (batched)
        field_ls = []
        image_ls = []
        single_rgb_dataset = TensorDataset(cat_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb.shape[1:]),
                dtype=self.dtype,
            )
        # print("batch size", _bs)

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            image, field = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            image_ls.append(image.detach())
            field_ls.append(field.detach())


        image_preds = torch.concat(image_ls, dim=0)
        field_preds = torch.concat(field_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------

        # Convert to numpy
        image_pred = image_preds.squeeze().cpu().permute(1,2,0).numpy()
        field_pred = field_preds.squeeze().cpu().permute(1,2,0).numpy()

        # normalize image and field
        image_pred = (image_pred + 1) * 255 / 2
        field_pred[:, :, 2] *= 90

        # Visualize; would need further work
        field_visualized =  draw_perspective_fields(image_pred, field_pred[:, :, :2], np.deg2rad(field_pred[:, :, 2]))

        return FinetuneOutput (
            image = Image.fromarray((image_pred).astype(np.uint8)),
            field = torch.tensor(field_pred), 
            field_visualized = Image.fromarray(field_visualized),
        )

    def _check_inference_step(self, n_step: int) -> None:
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(
                    f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps."
                )
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform an individual generation of field and image

        Args:
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = self.device
        rgb_in = rgb_in.to(device)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Initial map (noise)
        _rgb_in = rgb_in[:, :3, :]
        _field_in = rgb_in[:, 3:, :]
        cat_latent = self.encode_latent(_rgb_in, _field_in)
        cat_noise = torch.randn(
            cat_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 8, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (cat_noise.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            # predict the noise residual
            noise_pred = self.unet(
                cat_noise, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 8, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            cat_latent = self.scheduler.step(
                noise_pred, t, cat_latent, generator=generator
            ).prev_sample

        image, field = self.decode_latent(cat_latent) #flag

        return image, field

    def encode_latent(self, rgb_in: torch.Tensor, field_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image and corresponding field into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

            field_in (`torch.Tensor`)"
                Input field to be encoded.

        Returns:
            `torch.Tensor`: Image and field latent.
        """

        # normalize rgb
        rgb_norm: torch.Tensor = rgb_in / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0


        # encode rgb
        rgb_latent = self.vae.encoder(rgb_norm)
        moments = self.vae.quant_conv(rgb_latent)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale rgb latent
        rgb_latent = mean * self.rgb_latent_scale_factor

        # encode field
        field_latent = self.vae.encoder(field_in)
        moments = self.vae.quant_conv(field_latent)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        field_latent = mean 

        # concat the latents
        cat_latent = torch.cat([rgb_latent, field_latent], dim=1)

        return cat_latent

    def decode_latent(self, cat_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent into image and field

        Args:
            cat_latent (`torch.Tensor`):
                Concatenated latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """

        # might need to change later
        rgb_latent, field_latent = torch.tensor_split(cat_latent, 2, dim=1)

        # scale latent
        rgb_latent = rgb_latent / self.rgb_latent_scale_factor
        # decode
        z_rgb = self.vae.post_quant_conv(rgb_latent)
        rgb = self.vae.decoder(z_rgb)

        # decode
        z_field = self.vae.post_quant_conv(field_latent)
        field = self.vae.decoder(z_field)

        return (rgb, field)


    # Test whether we can decode a field latent
    def test_field_latent(self, rgb_in: torch.Tensor, field_in: torch.Tensor):
        # encode rgb

        # normalize rgb
        rgb_norm: torch.Tensor = rgb_in / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        rgb_latent = self.vae.encoder(rgb_norm)
        moments = self.vae.quant_conv(rgb_latent)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale rgb latent
        rgb_latent = mean * self.rgb_latent_scale_factor

        # encode field
        field_latent = self.vae.encoder(field_in)
        moments = self.vae.quant_conv(field_latent)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        field_latent = mean 


        # now, we decode then, to see if the decoder can recover the latents
        rgb_latent = rgb_latent / self.rgb_latent_scale_factor
        # decode
        z_rgb = self.vae.post_quant_conv(rgb_latent)
        rgb = self.vae.decoder(z_rgb)

        # decode
        z_field = self.vae.post_quant_conv(field_latent)
        field = self.vae.decoder(z_field)

        
        image_pred = rgb.squeeze().cpu().permute(1,2,0).numpy()
        field_pred = field.squeeze().cpu().permute(1,2,0).numpy()

        # normalize image and field
        image_pred = (image_pred + 1) * 255 / 2
        field_pred[:, :, 2] *= 90

        # Visualize; would need further work
        field_visualized =  draw_perspective_fields(image_pred, field_pred[:, :, :2], np.deg2rad(field_pred[:, :, 2]))

        '''

                # Test decoder output
                image_pred, vis_pred = self.model.test_field_latent(rgb[:1], field[:1])
                save_to_dir = os.path.join(
                self.out_dir_vis, self._get_backup_ckpt_name()
                )

                output_dir_jpg = os.path.join(save_to_dir, "image_test")
                output_dir_vis = os.path.join(save_to_dir, "field_visualization_test")
                os.makedirs(output_dir_jpg, exist_ok=True)
                os.makedirs(output_dir_vis, exist_ok=True)

                
                 # save image
                pred_name_base = "1_pred"
                jpg_save_path = os.path.join(output_dir_jpg, f"{pred_name_base}.jpg")
                if os.path.exists(jpg_save_path):
                    logging.warning(f"Existing file: '{jpg_save_path}' will be overwritten")
                image_pred.save(jpg_save_path)

                img_save_path = os.path.join(output_dir_jpg, f"{pred_name_base}_og.jpg")
                image = rgb[:1].squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8)
                

                # save visualized image
                vis_save_path = os.path.join(output_dir_vis, f"{pred_name_base}.jpg")
                if os.path.exists(vis_save_path):
                    logging.warning(f"Existing file: '{vis_save_path}' will be overwritten")
                vis_pred.save(vis_save_path)

                field = field[:1].squeeze().cpu().permute(1,2,0).numpy()

                field_visualized =  draw_perspective_fields(image, field[:, :, :2], np.deg2rad(field[:, :, 2] * 90))
                field_visualized = Image.fromarray(field_visualized)
                field_save_path_2 = os.path.join(output_dir_vis, f"{pred_name_base}_og.jpg")
                image = Image.fromarray(image)
                image.save(img_save_path)
                field_visualized.save(field_save_path_2)

            '''

        return (Image.fromarray(image_pred.astype(np.uint8)), Image.fromarray(field_visualized))
