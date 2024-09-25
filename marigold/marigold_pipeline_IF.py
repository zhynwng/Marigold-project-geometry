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


import html
import inspect
import re
import urllib.parse as ul
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    DiffusionPipeline,
    StableDiffusionMixin,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    IFPipeline,
    DDPMScheduler,
)
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.utils import (
    BaseOutput,
    BACKENDS_MAPPING,
    is_accelerate_available,
    is_accelerate_version,
    is_bs4_available,
    is_ftfy_available,
    logging,
    # randn_tensor,
    replace_example_docstring,
    is_torch_available,
    is_transformers_available,
)
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import pil_to_tensor, resize, to_pil_image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor#, T5EncoderModel, T5Toeknizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel
from diffusers.pipelines.deepfloyd_if.safety_checker import IFSafetyChecker
from diffusers.pipelines.deepfloyd_if.watermark import IFWatermarker
# from diffusers.pipelines.deepfloyd_if.pipeline_if_superresolution import super_resolution

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

# from .IF.deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
# from .IF.deepfloyd_if.modules.t5 import T5Embedder
# from .IF.deepfloyd_if.pipelines import super_resolution

# device = 'cuda:0'
# if_I = IFStageI('IF-I-M-v1.0', device=device)
# if_II = IFStageII('IF-II-M-v1.0', device=device)
# if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
# t5 = T5Embedder(device="cpu")

class DFIFOutput(BaseOutput):
    """
    Output class for IF pipeline that finetuned on perspective field

    Args:
        images (`List[Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        field (n.ndarray)
        field_visualized (Image.Image)
        nsfw_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content or a watermark. `None` if safety checking could not be performed.
        watermark_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely has a watermark. `None` if safety
            checking could not be performed.
    """

    image: Union[List[Image.Image], np.ndarray, Image.Image]
    field: np.ndarray
    field_visualized: Image.Image
    # nsfw_detected: Optional[List[bool]] # for IF
    # watermark_detected: Optional[List[bool]] # for IF

# try:
#     if not (is_transformers_available() and is_torch_available()):
#         raise OptionalDependencyNotAvailable()
# except OptionalDependencyNotAvailable:
#     from ...utils.dummy_torch_and_transformers_objects import *  # noqa F403
# else:
#     from .pipeline_if import IFPipeline
#     from .pipeline_if_img2img import IFImg2ImgPipeline
#     from .pipeline_if_img2img_superresolution import IFImg2ImgSuperResolutionPipeline
#     from .pipeline_if_inpainting import IFInpaintingPipeline
#     from .pipeline_if_inpainting_superresolution import IFInpaintingSuperResolutionPipeline
#     from .pipeline_if_superresolution import IFSuperResolutionPipeline
#     from .safety_checker import IFSafetyChecker
#     from .watermark import IFWatermarker


class DFIFPipeline(
    IFPipeline,
    DiffusionPipeline
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

    tokenizer: T5Tokenizer
    text_encoder: T5EncoderModel

    unet: UNet2DConditionModel
    scheduler: DDPMScheduler

    # feature_extractor: Optional[CLIPImageProcessor]
    # safety_checker: Optional[IFSafetyChecker]

    # watermarker: Optional[IFWatermarker]

    bad_punct_regex = re.compile(
        r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" + "\]" + "\[" + "\}" + "\{" + "\|" + "\\" + "\/" + "\*" + r"]{1,}"
    )  # noqa

    _optional_components = ["tokenizer", "text_encoder", "safety_checker", "feature_extractor", "watermarker"]

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
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        safety_checker: Optional[IFSafetyChecker],
        feature_extractor: Optional[CLIPImageProcessor],
        watermarker: Optional[IFWatermarker],
        requires_safety_checker: bool = False,

        # vae: AutoencoderKL,
        # image_encoder: CLIPVisionModelWithProjection = None,
        # feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
        scale_invariant: Optional[bool] = True,
        shift_invariant: Optional[bool] = True,
        default_denoising_steps: Optional[int] = None,
        default_processing_resolution: Optional[int] = None,
    ):
        super().__init__(tokenizer, text_encoder, unet, scheduler, safety_checker=None, feature_extractor=None, watermarker=None)

        # if safety_checker is None and requires_safety_checker:
        #     logger.warning(
        #         f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
        #         " that you abide to the conditions of the IF license and do not expose unfiltered"
        #         " results in services or applications open to the public. Both the diffusers team and Hugging Face"
        #         " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
        #         " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
        #         " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
        #     )

        # if safety_checker is not None and feature_extractor is None:
        #     raise ValueError(
        #         "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
        #         " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
        #     )

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            scheduler=scheduler,
            # safety_checker=safety_checker,
            # feature_extractor=feature_extractor,
            # watermarker=watermarker,
        )
        self.register_to_config(
            requires_safety_checker=requires_safety_checker
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
        # self,
        # input_field: Union[Image.Image, torch.Tensor, None],
        # denoising_steps: Optional[int] = None,
        # ensemble_size: int = 5,
        # processing_res: Optional[int] = None,
        # match_input_res: bool = True,
        # resample_method: str = "bilinear",
        # batch_size: int = 0,
        # generator: Union[torch.Generator, None] = None,
        # color_map: str = "Spectral",
        # show_progress_bar: bool = True,
        # ensemble_kwargs: Dict = None,

        self,
        input_field: Union[Image.Image, torch.Tensor, None],
        ensemble_size: int = 5,
        processing_res: Optional[int] = None,
        prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 100,
        batch_size: int = 0,
        resample_method: str = "bilinear",
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        show_progress_bar: bool = True,
    ) -> DFIFOutput:
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
        # if denoising_steps is None:
        #     denoising_steps = self.default_denoising_steps
        # if processing_res is None:
        #     processing_res = self.default_processing_resolution

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
                prompt="",
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                show_pbar=show_progress_bar,
                generator=generator,
            )
            image_ls.append(image_pred.detach())

        image_preds = torch.concat(image_ls, dim=0)
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------

        # Convert to numpy
        # image_pred = np.asarray(to_pil_image(image_preds[0]))
        image_pred = image_preds[0][:3,:,:].squeeze().cpu().permute(1,2,0).numpy()
        field_pred = input_field.squeeze().cpu().permute(1,2,0).numpy()

        # normalize image and field
        image_pred = (image_pred + 1) * 255 / 2
        field_pred[:, :, 2] *= 90 
        # image_pred = torchvision.transforms.functional.to_pil_image(image_pred)

        # Visualize; would need further work
        field_visualized =  draw_perspective_fields(image_pred, field_pred[:, :, :2], np.deg2rad(field_pred[:, :, 2] * 90))

        return DFIFOutput (
            image = Image.fromarray((image_pred).astype(np.uint8)),
            field = torch.tensor(field_pred), 
            field_visualized = Image.fromarray(field_visualized),
        )


    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        clean_caption: bool = False,
    ):
        """
        Encode text embedding for a given prompt
        """
        if prompt is not None and negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # while T5 can handle much longer input sequences than 77, the text encoder was trained with a max length of 77 for IF
        max_length = 77

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            attention_mask = text_inputs.attention_mask.to(device)

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.unet is not None:
            dtype = self.unet.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
        else:
            negative_prompt_embeds = None

        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds.to(device)

    def prepare_intermediate_images(self, batch_size, num_channels, height, width, dtype, device, generator):
        shape = (batch_size, num_channels, height, width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        intermediate_images = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        intermediate_images = intermediate_images * self.scheduler.init_noise_sigma
        return intermediate_images

    def prepare_extra_step_kwargs(self, generator, eta=0):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # @torch.no_grad()
    # def get_time_ids(self):
    #     '''
    #     Add time ids for the model
    #     '''

    #     device = self.device
    #     original_size = (1024, 1024)
    #     target_size = (1024, 1024)
    #     crops_coords_top_left = (0, 0)

    #     self.add_text_embeds = self.pooled_prompt_embeds.to(device)
    #     text_encoder_projection_dim = int(self.pooled_prompt_embeds.shape[-1])

    #     add_time_ids = list(original_size + crops_coords_top_left + target_size)

    #     passed_add_embed_dim = (
    #         self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
    #     )
    #     expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

    #     if expected_add_embed_dim != passed_add_embed_dim:
    #         raise ValueError(
    #             f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
    #         )

    #     self.add_time_ids = torch.tensor([add_time_ids], dtype=self.prompt_embeds.dtype).to(device)


    @torch.no_grad()
    def single_infer(
        self,
        field_in: torch.Tensor,
        prompt: str,
        num_inference_steps: int,
        generator: Union[torch.Generator, None],
        show_pbar: bool,
        num_images_per_prompt: Optional[int] = 1,
        timesteps: List[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clean_caption: bool = False,        
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
        # field_latent = self.encode_field(field_in)

        # Define call parameters
        # print("self unet config sample size", self.unet.config.sample_size)
        height = height or self.unet.config.sample_size
        width = width or self.unet.config.sample_size

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # do_classifier_free_guidance = guidance_scale > 1.0

        # Set timesteps
        if timesteps is not None:
            self.scheduler.set_timesteps(timesteps=timesteps, device=device)
            timesteps = self.scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)

        # print("unet config class embed type", self.unet.config.class_embed_type)
        # self.unet.config.class_labels_conditioning = "timestep"

        #prepare latents
        # latents = torch.randn(field_latent.shape, generator=generator, device=device, dtype=self.prompt_embeds.dtype)
        # scale the initial noise by the standard deviation required by the scheduler
        # latents = latents * self.scheduler.init_noise_sigma

        # Encode prompt
        if self.prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                do_classifier_free_guidance,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                clean_caption=clean_caption,
            )

        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        # self.add_text_embeds = self.add_text_embeds.to(device)
        # self.add_time_ids = self.add_time_ids.to(device)
        self.prompt_embeds = self.prompt_embeds.to(device)

        # Prepare intermediate images
        intermediate_images = self.prepare_intermediate_images(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height,
            width,
            self.prompt_embeds.dtype,
            device,
            generator,
        )

        # Added: Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator)#, eta)
        # print('extra_step_kwargs',extra_step_kwargs)

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in iterable:
            # expand if we are doing classifier free guidance
            # latent_model_input = torch.cat([field_latent, latents], dim=1)
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            intermediate_input = torch.cat([field_in, intermediate_images], dim=1)
            model_input = (
                intermediate_input # torch.cat([intermediate_images] * 2) if do_classifier_free_guidance else intermediate_images
            )
            model_input = self.scheduler.scale_model_input(model_input, t)
            # print('sch.tm',self.scheduler.timesteps.shape)
            # print('prompt embd',self.prompt_embeds.shape)
            # predict the noise residual
            noise_pred = self.unet(
                model_input,
                t,
                class_labels=torch.Tensor([0]).to(device),#self.scheduler.timesteps,
                encoder_hidden_states=self.prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # # perform guidance
            # if do_classifier_free_guidance:
            #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            #     noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            #     noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            #     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            #     noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)
            # print('noise_pred',noise_pred.shape,'intermediate_images',intermediate_images.shape)
            # compute the previous noisy sample x_t -> x_t-1
            # intermediate_images = self.scheduler.step(
            #     noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
            # )[0]

        # high_res = super_resolution(
        #             t5,
        #             if_III=if_III,
        #             prompt=[''],
        #             support_pil_img=middle_res['III'][0],
        #             img_scale=4.,
        #             img_size=256,
        #             if_III_kwargs={
        #                 "guidance_scale": 9.0,
        #                 "noise_level": 20,
        #                 "sample_timestep_respacing": "75",
        #             },
        #         )

        image = intermediate_images

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

# def super_resolution(
#     t5,
#     if_I=None,
#     *,
#     support_pil_img,
#     prompt=None,
#     negative_prompt=None,
#     seed=None,
#     if_I_kwargs=None,
#     progress=True,
#     img_size=256,
#     img_scale=4.0,
#     return_tensors=False,
#     disable_watermark=False,
# ):
#     assert isinstance(support_pil_img, PIL.Image.Image)
#     assert img_size % 8 == 0

#     if seed is None:
#         seed = int((datetime.utcnow().timestamp() * 10 ** 6) % (2 ** 32 - 1))

#     if prompt is not None:
#         t5_embs = t5.get_text_embeddings(prompt)
#     else:
#         t5_embs = t5.get_text_embeddings('')

#     if negative_prompt is not None:
#         if isinstance(negative_prompt, str):
#             negative_prompt = [negative_prompt]
#         negative_t5_embs = t5.get_text_embeddings(negative_prompt)
#     else:
#         negative_t5_embs = None

#     low_res = _prepare_pil_image(support_pil_img, img_size)

#     result = {}

#     bs = 1
#     if_I_kwargs = if_I_kwargs or {}

#     if if_I.use_diffusers:
#         if_I_kwargs['prompt'] = prompt

#     if_I_kwargs['low_res'] = low_res.repeat(bs, 1, 1, 1)
#     if_I_kwargs['seed'] = seed
#     if_I_kwargs['t5_embs'] = t5_embs
#     if_I_kwargs['negative_t5_embs'] = negative_t5_embs
#     if_I_kwargs['progress'] = progress
#     if_I_kwargs['img_scale'] = img_scale

#     stageI_generations, _meta = if_I.embeddings_to_image(**if_I_kwargs)
#     pil_images_I = if_I.to_images(stageI_generations, disable_watermark=disable_watermark)
#     result['I'] = pil_images_I

#     if return_tensors:
#         return result, (stageI_generations,)
#     else:
#         return result