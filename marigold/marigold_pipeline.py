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
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
    LMSDiscreteScheduler
)
from diffusers.utils import BaseOutput
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .util.batchsize import find_batch_size
from .util.ensemble import ensemble_depth
from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)

from perspective2d.utils import draw_perspective_fields

class MarigoldOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    image: Image.Image


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

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

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[LMSDiscreteScheduler],
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

    @torch.no_grad()
    def __call__(
        self,
        input_object: torch.Tensor ,
        input_shadow: torch.Tensor, 
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
    ) -> MarigoldOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_rgb (`torch.Tensor`):
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
        self._check_inference_step(denoising_steps)

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Convert to torch tensor
        input_size = input_shadow.shape
        assert (
            4 == input_shadow.dim() and 3 == input_size[-3]
        ), f"Wrong input shape {input_size}, expected [1, rgb, H, W]"

        # Resize image
        if processing_res > 0:
            input_shadow = resize_max_res(
                input_shadow,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # ----------------- Predicting image -----------------
        # Batch repeated input image
        duplicated_shadow = input_shadow.expand(ensemble_size, -1, -1, -1)
        duplicated_object = input_object.expand(ensemble_size, -1, -1, -1)
        single_OS_dataset = TensorDataset(duplicated_object, duplicated_shadow)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_OS_loader = DataLoader(
            single_OS_dataset, batch_size=_bs, shuffle=False
        )

        # Predict image (batched)
        rgb_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_OS_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_OS_loader
        for batch in iterable:
            (batched_object, batched_shadow, ) = batch
            rgb_pred_raw = self.single_infer(
                object_in=batched_object,
                shadow_in=batched_shadow,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            rgb_pred_ls.append(rgb_pred_raw.detach())
        rgb_preds = torch.concat(rgb_pred_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # Convert to numpy
        rgb_pred = torch.clip(rgb_preds, -1.0, 1.0)
        rgb_pred = ((rgb_pred + 1.0) / 2.0).squeeze()
        rgb = (rgb_pred.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        
        return MarigoldOutput (
            image = Image.fromarray(rgb),
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

    def encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = "a realistic photo of an indoor room"
        text_inputs = self.tokenizer(prompt, 
            padding="max_length", 
            max_length= self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt")

        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        object_in: torch.Tensor,
        shadow_in: torch.Tensor, 
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            object_in (`torch.Tensor`):
                Input object image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
            generator (`torch.Generator`)
                Random generator for initial noise generation.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """

        guidance_scale = 7.5
        device = self.device
        object_in = object_in.to(device)
        shadow_in = shadow_in.to(device)


        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode OS
        obj_norm: torch.Tensor = object_in / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        obj_norm = obj_norm.float()
        assert obj_norm.min() >= -1.0 and obj_norm.max() <= 1.0

        shadow_norm: torch.Tensor = shadow_in / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        shadow_norm = obj_norm.float()
        assert shadow_norm.min() >= -1.0 and shadow_norm.max() <= 1.0


        # Encode object shadow
        obj_latent = self.encode_rgb(obj_norm)  # [B, 4, h, w]
        shadow_latent = self.encode_rgb(shadow_norm)  # [B, 4, h, w]

        # Initial image latent (noise)
        rgb_latent = torch.randn(
            obj_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        ).to(device)  # [B, 2, 1024]


        # guidance embedding
        uncond_input = self.tokenizer(
            [""] * rgb_latent.shape[0], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]  
        text_embeddings = torch.cat([uncond_embeddings, batch_empty_text_embed])

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

        rgb_latent = rgb_latent * self.scheduler.init_noise_sigma

        for i, t in iterable:            
            unet_input = torch.cat(
                [obj_latent, shadow_latent, rgb_latent], dim=1
            )  # this order is important

            # guidance
            unet_input = torch.cat([unet_input] * 2)
            unet_input =  self.scheduler.scale_model_input(unet_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=text_embeddings
            ).sample  # [B, 4, h, w]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            rgb_latent = self.scheduler.step(
                noise_pred, t, rgb_latent
            ).prev_sample
            

        rgb = self.decode_rgb(rgb_latent)

        return rgb

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """

    
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_rgb(self, rgb_latent: torch.Tensor) -> torch.Tensor:

        rgb_latent = 1 / self.rgb_latent_scale_factor * rgb_latent
        # decode
        z = self.vae.post_quant_conv(rgb_latent)
        rgb = self.vae.decoder(z)

        return rgb

    def encode_field(self, field_in: torch.Tensor) -> torch.Tensor:
        # encode
        h = self.vae.encoder(field_in)
        moments = self.vae.quant_conv(h)
        field_latent, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        return field_latent

    
    def decode_field(self, field_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode field latent into field map.

        Args:
            field_latent (`torch.Tensor`):
                Field latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # decode
        z = self.vae.post_quant_conv(field_latent)
        field = self.vae.decoder(z)

        return field

