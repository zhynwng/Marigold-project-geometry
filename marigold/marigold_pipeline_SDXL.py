# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-05-24
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
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
from typing import Dict, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    DiffusionPipeline,
    StableDiffusionMixin,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.utils import BaseOutput
from PIL import Image
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)

from perspective2d.utils import draw_perspective_fields
import matplotlib.pyplot as plt

class SDXLOutput(BaseOutput):
    """
    Output class for SDXL pipeline that finetuned on perspective field

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    image: Image.Image
    field: np.ndarray
    field_visualized: Image.Image


class SDXLPipeline(
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    StableDiffusionMixin,
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
    ):
    """
    Pipeline for SDXL model finetuned on perspective field

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
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

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        # self,
        # unet: UNet2DConditionModel,
        # vae: AutoencoderKL,
        # scheduler: Union[DDIMScheduler, LCMScheduler],
        # text_encoder: CLIPTextModel,
        # tokenizer: CLIPTokenizer,
        # scale_invariant: Optional[bool] = True,
        # shift_invariant: Optional[bool] = True,
        # default_denoising_steps: Optional[int] = None,
        # default_processing_resolution: Optional[int] = None,

        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: EulerDiscreteScheduler,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__(vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            scale_invariant=scale_invariant,
            shift_invariant=shift_invariant,
            default_denoising_steps=default_denoising_steps,
            default_processing_resolution=default_processing_resolution,
        )

        self.scale_invariant = scale_invariant
        self.shift_invariant = shift_invariant
        self.default_denoising_steps = default_denoising_steps
        self.default_processing_resolution = default_processing_resolution

        self.prompt_embeds = None
        self.pooled_prompt_embeds = None
        self.add_time_ids = None
        self.add_text_embeds = None

        self.unet.to(self.device)


    @torch.no_grad()
    def __call__(
        self,
        input_field: Union[Image.Image, torch.Tensor, None],
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
    ) -> SDXLOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`torch.Tensor`):
                Input image.
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
        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Field Preprocess -----------------
        input_size = input_field.shape # [B, C, h, w]
        assert (
            4 == input_field.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            input_field = resize_max_res(
                input_field,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # ----------------- Predicting images -----------------
        image_ls = []
        # Batch repeated input image
        duplicated_field = input_field.expand(ensemble_size, -1, -1, -1) # [B, C, 256, 256]
        single_field_dataset = TensorDataset(duplicated_field)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )
        single_field_loader = DataLoader(
            single_field_dataset, batch_size=_bs, shuffle=False
        )

        # Predict perspective field maps (batched)
        field_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_field_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_field_loader
        for batch in iterable:
            (batched_field,) = batch
            image_pred = self.single_infer(
                field_in=batched_field,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            image_ls.append(image_pred.detach())

        image_preds = torch.concat(image_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------

        # Convert to numpy
        image_pred = image_preds[0].squeeze().cpu().permute(1,2,0).numpy()
        field_pred = input_field.squeeze().cpu().permute(1,2,0).numpy()

        # normalize image and field
        image_pred = (image_pred + 1) * 255 / 2
        field_pred[:, :, 2] *= 90 

        # Visualize; would need further work
        field_visualized =  draw_perspective_fields(image_pred, field_pred[:, :, :2], np.deg2rad(field_pred[:, :, 2] * 90))

        return SDXLOutput (
            image = Image.fromarray((image_pred).astype(np.uint8)),
            field = torch.tensor(field_pred), 
            field_visualized = Image.fromarray(field_visualized),
        )


    @torch.no_grad()
    def encode_prompt(self):
        """
        Encode text embedding for a given prompt
        """
        num_images_per_prompt = 1
        device = self.device
        prompt = "a realistic picture of an indoor room"
        prompt = [prompt]
        batch_size = 1

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2]
        )

        prompt_2 = prompt
        # textual inversion: process multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length= tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])

            prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]

            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)


        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        self.prompt_embeds = prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds.to(device)



    @torch.no_grad()
    def get_time_ids(self):
        '''
        Add time ids for the model
        '''

        device = self.device
        original_size = (1024, 1024)
        target_size = (1024, 1024)
        crops_coords_top_left = (0, 0)

        self.add_text_embeds = self.pooled_prompt_embeds.to(device)
        text_encoder_projection_dim = int(self.pooled_prompt_embeds.shape[-1])

        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        self.add_time_ids = torch.tensor([add_time_ids], dtype=self.prompt_embeds.dtype).to(device)


    @torch.no_grad()
    def single_infer(
        self,
        field_in: torch.Tensor,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            field_in (`torch.Tensor`):
                Input RGB image.
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
        batch_size = field_in.shape[0]

        # encode field 
        field_latent = self.encode_field(field_in)


        # Set time steps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        #prepare latents
        latents = torch.randn(field_latent.shape, generator=generator, device=device, dtype=self.prompt_embeds.dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma


        if self.prompt_embeds is None:
            self.encode_prompt()
            self.get_time_ids()

        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        self.add_text_embeds = self.add_text_embeds.to(device)
        self.add_time_ids = self.add_time_ids.to(device)
        self.prompt_embeds = self.prompt_embeds.to(device)


        for i, t in iterable:
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([field_latent, latents], dim=1)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": self.add_text_embeds, "time_ids": self.add_time_ids}


            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states= self.prompt_embeds,
                cross_attention_kwargs= None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

        image = self.decode_rgb(latents)

        return image



    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        latent = self.vae.encode(rgb_in).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor

        latent = latent.to(self.device)

        return latent


    def encode_field(self, field_in: torch.Tensor) -> torch.Tensor:
        latent = self.vae.encode(field_in).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
    
        latent = latent.to(self.device)

        return latent


    def decode_rgb(self, latents: torch.Tensor) -> torch.Tensor:
        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.vae.config.scaling_factor

        image = self.vae.decode(latents, return_dict=False)[0]

        return image
