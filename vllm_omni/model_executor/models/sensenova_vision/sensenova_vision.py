# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SenseNova-Vision-7B-MoT omni model.

SenseNova-Vision is a fork of Bagel with identical parameter-bearing modules.
This class reuses the MoT/ViT/VAE embedding logic from the BAGEL integration
and only overrides the SenseNovaVision checkpoint defaults plus additive features
(e.g. ``return_raw_latent``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

# from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY

# from vllm.multimodal.inputs import MultiModalKwargsItems
# from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems, MultiModalDataItems
# from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails
from vllm_omni.model_executor.models.bagel.bagel import (
    OmniBagelDummyInputsBuilder,
    OmniBagelForConditionalGeneration,
    OmniBagelMultiModalProcessor,
    OmniBagelProcessingInfo,
    OmniBagelProcessor,
)

logger = init_logger(__name__)

# Official SenseNova-Vision VAE image transform, transcribed from the upstream
# ``ImageTransform(1024, 512, 16)`` (``sensenova_vision.py`` ``vae_transform``).
# ``ImageTransform`` applies a ``MaxLongEdgeMinShortEdgeResize``: the long edge
# is scaled down to at most ``max_size``, the short edge scaled up to at least
# ``min_size``, and the result rounded to a multiple of ``stride``.
#
# For non-recon3d img2img/mixed modes the AR stage conditions the input
# image through this transform and caches the result in
# ``kv_metadata["image_shape"]``, so the DiT grid matches the official
# pipeline instead of BAGEL's hardcoded short-edge floor of 256.  The same
# dims feed both the AR model resize (``_resize_to_stride`` override) and the
# processor's VAE placeholder sizing, keeping them in lockstep.
SENSENOVA_VISION_VAE_MAX_SIZE = 1024
SENSENOVA_VISION_VAE_MIN_SIZE = 512
SENSENOVA_VISION_VAE_STRIDE = 16

# Official SenseNova-Vision ViT image transform (``ImageTransform(980, 224, 14)``):
# aspect-PRESERVING ``MaxLongEdgeMinShortEdgeResize`` -- long edge <= 980, short
# edge >= 224, rounded to stride 14, pixel budget 14*14*9*1024/img.  Upstream
# runs this on the VAE-resized image inside ``update_context_image``, so the
# ViT patch count FOLLOWS the aspect ratio instead of being a fixed square.
SENSENOVA_VISION_VIT_MAX_SIZE = 980
SENSENOVA_VISION_VIT_MIN_SIZE = 224
SENSENOVA_VISION_VIT_STRIDE = 14
SENSENOVA_VISION_VIT_MAX_PIXELS = 14 * 14 * 9 * 1024


def _sensenova_vae_resize_dims(img_h: int, img_w: int) -> tuple[int, int]:
    """Stride-aligned ``(new_h, new_w)`` for the SenseNova-Vision VAE transform.

    Ports ``MaxLongEdgeMinShortEdgeResize`` with ``(max=1024, min=512, stride=16)``
    so the AR VAE grid / `image_shape` matches the official pipeline.  Used by
    both the AR model resize and the processor's placeholder sizing so they
    never diverge.
    """
    stride = SENSENOVA_VISION_VAE_STRIDE
    max_size = SENSENOVA_VISION_VAE_MAX_SIZE
    min_size = SENSENOVA_VISION_VAE_MIN_SIZE

    scale = min(max_size / max(img_h, img_w), 1.0)
    scale = max(scale, min_size / min(img_h, img_w))
    new_h = max(stride, int(round(img_h * scale / stride) * stride))
    new_w = max(stride, int(round(img_w * scale / stride) * stride))
    if max(new_h, new_w) > max_size:
        scale = max_size / max(new_h, new_w)
        new_h = max(stride, int(round(new_h * scale / stride) * stride))
        new_w = max(stride, int(round(new_w * scale / stride) * stride))
    return new_h, new_w


# Per-task image target side for the generation output grid.  Recon3D decodes
# ``num_views`` square views at this VAE side; the AR stage caches the same
# value in ``kv_metadata["image_shape"]`` so the DiT stage (SenseNovaVisionPipeline)
# and the AR prefill agree on the latent grid.
RECON3D_VAE_SIDE = 512

# ``num_output_vae`` in upstream ``gen_image`` (inferencer.py) when no explicit
# per-request ``num_views`` is supplied.
RECON3D_DEFAULT_NUM_VIEWS = 4


def _sensenova_make_divisible(value: int, stride: int) -> int:
    """Mirror ``MaxLongEdgeMinShortEdgeResize._make_divisible``."""
    return max(stride, int(round(value / stride)) * stride)


def _sensenova_vit_resize_dims(vae_h: int, vae_w: int) -> tuple[int, int]:
    """Stride-aligned ``(vit_h, vit_w)`` for ``ImageTransform(980, 224, 14)``.

    Ports ``MaxLongEdgeMinShortEdgeResize`` for the ViT branch (bicubic,
    antialias default), including the max-pixels shrink and the final
    longest-edge cap.  Input is the ALREADY-VAE-RESIZED image, matching
    upstream ``interleave_inference`` (the ViT transform is applied to the
    vae-transformed image).  Must be kept in lockstep with
    ``_process_img2img_input`` so the placeholder count equals the patch
    count.
    """
    max_size = SENSENOVA_VISION_VIT_MAX_SIZE
    min_size = SENSENOVA_VISION_VIT_MIN_SIZE
    stride = SENSENOVA_VISION_VIT_STRIDE

    def apply_scale(width: int, height: int, scale: float) -> tuple[int, int]:
        return (
            _sensenova_make_divisible(round(width * scale), stride),
            _sensenova_make_divisible(round(height * scale), stride),
        )

    scale = min(max_size / max(vae_h, vae_w), 1.0)
    scale = max(scale, min_size / min(vae_h, vae_w))
    vit_w, vit_h = apply_scale(vae_w, vae_h, scale)

    if vit_w * vit_h > SENSENOVA_VISION_VIT_MAX_PIXELS:
        shrink = SENSENOVA_VISION_VIT_MAX_PIXELS / (vit_w * vit_h)
        vit_w, vit_h = apply_scale(vit_w, vit_h, shrink)
    if max(vit_w, vit_h) > max_size:
        shrink = max_size / max(vit_w, vit_h)
        vit_w, vit_h = apply_scale(vit_w, vit_h, shrink)
    return vit_h, vit_w


# SenseNova-Vision-7B-MoT defaults.  The checkpoint ships metadata-only
# ``config.json`` (no ``architectures``), so these constants mirror what the
# official ``SenseNovaVisionModel._build_model`` applies at load time
# (inference/sensenova_vision.py).
SENSENOVA_VISION_DEFAULT_LAYER_MODULE = "Qwen2MoTDecoderLayer"
SENSENOVA_VISION_DEFAULT_QK_NORM = True
SENSENOVA_VISION_DEFAULT_TIE_WORD_EMBEDDINGS = False
SENSENOVA_VISION_DEFAULT_VISUAL_GEN = True
SENSENOVA_VISION_DEFAULT_VISUAL_UND = True
SENSENOVA_VISION_DEFAULT_MAX_LATENT_SIZE = 64
SENSENOVA_VISION_DEFAULT_VIT_MAX_NUM_PATCH_PER_SIDE = 70


def _sensenova_vit_patch_count(vae_h: int, vae_w: int) -> int:
    """Aspect-aware ViT patch count for a VAE-RESIZED image.

    The caller already applied the VAE transform; this applies only the ViT
    transform ``ImageTransform(980, 224, 14)`` and counts the patches.  Used by
    both the processor's placeholder sizing and the AR model's
    ``_encode_vit_embeddings`` so they never diverge (the previous version
    redundantly re-ran the VAE resize here, which the model side never does in
    ``_encode_vit_embeddings`` - it only consumes the already-VAE-resized
    image).
    """
    vit_h, vit_w = _sensenova_vit_resize_dims(int(vae_h), int(vae_w))
    return (vit_h // SENSENOVA_VISION_VIT_STRIDE) * (vit_w // SENSENOVA_VISION_VIT_STRIDE)


def _sensenova_understanding_patch_count(img_h: int, img_w: int) -> int:
    """Aspect-aware ViT patch count for an ORIGINAL understanding image.

    Mirrors the model-side understanding embedding chain (``_resize_to_stride``
    then ``_encode_vit_embeddings``): the full VAE transform is applied first,
    then the ViT transform to the VAE-resized image.  This equals the number of
    ``<|image_pad|>`` placeholder tokens per understanding image.
    """
    vae_h, vae_w = _sensenova_vae_resize_dims(int(img_h), int(img_w))
    return _sensenova_vit_patch_count(vae_h, vae_w)


class OmniSenseNovaVisionProcessor(OmniBagelProcessor):
    """SenseNovaVision image processor: pass original-res pixels to the AR model.

    Upstream ``InterleaveInferencer.interleave_inference`` resizes the ORIGINAL
    image with the VAE transform (``ImageTransform(1024, 512, 16)``) before the
    ViT transform, and never squashes it to a fixed square.  The base
    ``OmniBagelProcessor`` disables ``do_resize`` for the ``img2img`` modality;
    SenseNovaVision needs the same for the plain ``image`` (understanding)
    modality so the patch count follows the aspect ratio and stays in lockstep
    with ``_process_img2text_input``.
    """

    # transformers>=5.0 ProcessorMixin.get_attributes() only scans the LEAF
    # class's __dict__ for ``<attribute>_class`` hints. Since this class is
    # what from_pretrained() instantiates, redeclare the hints so
    # ``self.image_processor`` / ``self.tokenizer`` are set (the parent
    # OmniBagelProcessor redeclarations are invisible to the leaf scan).
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __call__(self, text=None, images=None, **kwargs):
        is_img2text = not kwargs.pop("is_img2img", False)

        # if images is not None and is_img2text:
        #     # Raw (aspect-preserving) pixels: no square pre-resize.  Mirror the
        #     # base img2img raw branch; the orchestrating ``_call_hf_processor``
        #     # keeps the image/img2img key names distinct.
        #     from vllm.transformers_utils.processors.bagel import BagelProcessorKwargs

        #     output_kwargs = self._merge_kwargs(
        #         BagelProcessorKwargs,
        #         tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        #         **kwargs,
        #     )
        #     image_kwargs = dict(output_kwargs["images_kwargs"])
        #     image_kwargs["do_resize"] = False
        #     image_kwargs["do_rescale"] = True
        #     image_kwargs.setdefault("return_tensors", "pt")
        #     pixel_values = self.image_processor(images, **image_kwargs)

        #     text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"]) if text is not None else None

        #     if pixel_values is not None and text_inputs is not None:
        #         combined = dict(text_inputs)
        #         combined["pixel_values"] = pixel_values["pixel_values"]
        #         return BatchFeature(combined)
        #     elif pixel_values is not None:
        #         return pixel_values
        #     elif text_inputs is not None:
        #         return BatchFeature(dict(text_inputs))
        #     else:
        #         return BatchFeature({})

        return super().__call__(text, images, is_img2img=not is_img2text, **kwargs)


class OmniSenseNovaVisionProcessingInfo(OmniBagelProcessingInfo):
    """Multi-modal limits for SenseNova-Vision.

    The shared BAGEL base caps ``image`` / ``img2img`` at 1 item per request;
    SenseNova-Vision raises both to 10, matching the upstream recon3d
    ``max_images=10``.  A finite cap (never ``None``) keeps mm memory
    profiling bounded.
    """

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 10, "img2img": 10}

    def get_hf_processor(self, **kwargs: object):
        # Raw (aspect-preserving) pixels for BOTH image and img2img modalities
        # so the AR model can apply the official VAE->ViT resize chain.
        return self.ctx.get_hf_processor(OmniSenseNovaVisionProcessor, **kwargs)


class OmniSenseNovaVisionMultiModalProcessor(OmniBagelMultiModalProcessor):
    """SenseNovaVision multimodal processor with recon3d view plumbing.

    Subclasses :class:`OmniBagelMultiModalProcessor` additively: it passes
    ``target_h``/``target_w`` through for the generation ``image`` modality so a
    recon3 request can request the VAE target size per view, and recomputes the
    ``img2img`` VAE placeholder count with the official
    ``ImageTransform(1024, 512, 16)`` short-edge floor so it stays in lockstep
    with the AR model's VAE latent grid.
    """

    def _mm_kwargs_for_bagel_img2img_hf(self, mm_kwargs):
        # SenseNovaVision recon3d views are decoded additively from the AR KV
        # cache; ``target_h``/``target_w`` select the VAE output side and are
        # preserved through to the ``image_shape`` used by the DiT stage.
        return dict(mm_kwargs)

    # def _get_prompt_updates(
    #     self,
    #     mm_items: MultiModalDataItems,
    #     hf_processor_mm_kwargs: Mapping[str, object],
    #     out_mm_kwargs: MultiModalKwargsItems,
    # ) -> list[PromptReplacement]:
    #     """Placeholder sizing for understanding images, in lockstep with the model.

    #     The user's img2text flow feeds the AR model through
    #     ``_process_img2text_input`` -> ``_resize_to_stride`` (VAE transform) ->
    #     ``_encode_vit_embeddings`` (ViT transform), whose patch count is
    #     ``_sensenova_vit_patch_count(_sensenova_vae_resize_dims(h, w))``.  The
    #     BAGEL base counts with ``bagel_image_size`` applied DIRECTLY to the
    #     original image, which can diverge for aspect ratios where the VAE
    #     short-edge floor changes the shape before the ViT step (e.g. extreme
    #     panoramas) -- an off-by-one here turns into an embedding/placeholder
    #     count mismatch.  Keep the base img2img replacement (the img2img model
    #     path resizes with ``bagel_image_size`` directly and stays consistent)
    #     and only swap the plain ``image`` modality to the same count the AR
    #     model produces.
    #     """
    #     replacements = super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs)
    #     tokenizer = self.info.get_tokenizer()
    #     image_token_id = tokenizer.get_vocab().get("<|image_pad|>")
    #     if image_token_id is None:
    #         return replacements

    #     def understanding_replacement(item_idx: int):
    #         size = mm_items.get_items("image", ImageProcessorItems).get_image_size(item_idx)
    #         count = _sensenova_understanding_patch_count(int(size.height), int(size.width)) + 2
    #         return [image_token_id] * count

    #     return [
    #         PromptReplacement(
    #             modality="image",
    #             target=[image_token_id],
    #             replacement=understanding_replacement,
    #         )
    #         if r.modality == "image"
    #         else r
    #         for r in replacements
    #     ]


@MULTIMODAL_REGISTRY.register_processor(
    OmniSenseNovaVisionMultiModalProcessor,
    info=OmniSenseNovaVisionProcessingInfo,
    dummy_inputs=OmniBagelDummyInputsBuilder,
)
class OmniSenseNovaVisionForConditionalGeneration(OmniBagelForConditionalGeneration):
    """SenseNova-Vision-7B-MoT omni model (subclass of the BAGEL integration).

    Inherits the entire MoT/ViT/VAE embedding and KV-transfer logic from
    :class:`OmniBagelForConditionalGeneration`.  Only the checkpoint-specific
    defaults differ:

    - ``layer_module="Qwen2MoTDecoderLayer"``
    - ``qk_norm=True``
    - ``tie_word_embeddings=False``
    - ``visual_gen=True`` / ``visual_und=True``
    - ``max_latent_size=64`` (BAGEL ships 32)
    - ``vit_max_num_patch_per_side=70``

    Additive SenseNovaVision features (``return_raw_latent`` for the diffusion
    pipeline) are exposed here without forking the base implementation.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Make AutoTokenizer.from_pretrained resolve VLLMSenseNovaVisionTokenizer
        # (id-preserving) in this process BEFORE the BAGEL core builds its own
        # tokenizer at OmniBagelForConditionalGeneration.__init__ (bagel.py) and
        # derives the img2img marker ids. Without registration the checkpoint's
        # declared tokenizer_class cannot be resolved from remote code (the
        # tokenization_sensenova_vision.py source file is never written into the
        # checkpoint dir), so AutoTokenizer would renumber the added tokens past
        # the 152064 embedding rows and trip the embed gather device-side assert.
        from vllm_omni.diffusion.models.sensenova_vision.tokenization_sensenova_vision import (
            register_vllm_sensenova_vision_tokenizer,
        )

        register_vllm_sensenova_vision_tokenizer()
        config = vllm_config.model_config.hf_config
        self._apply_sensenova_vision_config_defaults(config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Token id of the plain understanding image placeholder.  The base
        # derives the vision-marker / fim ids from the registered tokenizer;
        # the understanding path needs this id for the position rewrite.
        tok = getattr(self, "_probe_tokenizer", None) or getattr(self, "tokenizer", None)
        if tok is not None:
            self._img2text_token_id = int(tok.convert_tokens_to_ids("<|image_pad|>"))
        else:
            self._img2text_token_id = -1

        # Per-request img2img layout state machine.  Mirrors what
        # the single-image base keeps in ``_pending_img2img_info``/``_last`` but
        # keyed by req_id so continuation chunks and split blocks survive
        # across forward() calls.  See _adjust_positions_for_img2img.
        self._img2img_layouts: dict[str, dict[str, object]] = {}
        # Per-step schedule (req_id, num_computed_tokens, num_scheduled_tokens)
        # captured in batch order by prepare_runner_inputs.  The AR runner
        # already provides these per-request tensors with NO core runner
        # change; there is deliberately no ``num_prompt_tokens`` channel (the
        # reverted runner hook), so the state machine must not depend on one.
        self._step_req_schedule: list[tuple[str, int, int]] = []

    @staticmethod
    def _apply_sensenova_vision_config_defaults(config) -> None:
        """Force SenseNovaVision checkpoint defaults on the HF config in place."""
        config.visual_gen = SENSENOVA_VISION_DEFAULT_VISUAL_GEN
        config.visual_und = SENSENOVA_VISION_DEFAULT_VISUAL_UND
        config.max_latent_size = SENSENOVA_VISION_DEFAULT_MAX_LATENT_SIZE
        config.vit_max_num_patch_per_side = SENSENOVA_VISION_DEFAULT_VIT_MAX_NUM_PATCH_PER_SIDE

        llm_config = config.llm_config
        llm_config.layer_module = SENSENOVA_VISION_DEFAULT_LAYER_MODULE
        llm_config.qk_norm = SENSENOVA_VISION_DEFAULT_QK_NORM
        llm_config.tie_word_embeddings = SENSENOVA_VISION_DEFAULT_TIE_WORD_EMBEDDINGS

    def get_raw_latent(self) -> None:
        """SenseNovaVision additive feature: raw-latent flag extension point.

        The AR stage produces KV caches, never latents, so this returns
        ``None``.  The DiT stage (``SenseNovaVisionPipeline``) honors
        ``return_raw_latent`` when decoding images.
        """
        return None

    # def _resize_to_stride(self, pixel_values: torch.Tensor) -> torch.Tensor:
    #     """Resize img2img pixel values to the official SenseNova-Vision VAE grid.

    #     Overrides the BAGEL base (whose short-edge floor is ``min(256, max)``)
    #     with the official ``ImageTransform(1024, 512, 16)``: the long edge is
    #     clamped to 1024 and the short edge pushed up to at least 512.  The
    #     base ``_process_img2img_input`` calls this per image, so the resulting
    #     ``image_shape`` cached in ``kv_metadata["image_shape"]`` lands the DiT
    #     output at the official resolution (e.g. a (375, 500) input becomes
    #     (368, 496) -> 512-aligned (512, 688)).
    #     """
    #     H, W = pixel_values.shape[2], pixel_values.shape[3]
    #     new_H, new_W = _sensenova_vae_resize_dims(H, W)
    #     if new_H != H or new_W != W:
    #         pixel_values = torch.nn.functional.interpolate(
    #             pixel_values, size=(new_H, new_W), mode="bicubic", align_corners=False
    #         )
    #     return pixel_values

    def _clear_warmup_state(self):
        """Clear stale state accumulated during warmup/profiling runs."""
        super()._clear_warmup_state()
        self._img2img_layouts.clear()
        self._step_req_schedule.clear()

    def prepare_runner_inputs(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        req_ids: Sequence[str],
        num_computed_tokens: Sequence[int],
        num_scheduled_tokens: Sequence[int],
        input_ids_buffer: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Restore input_ids and capture the per-request step schedule.

        Mirrors the BAGEL base (restores ``input_ids`` from
        ``input_ids_buffer`` so the position rewrite can locate the
        ``<|vision_start|>`` block) and additionally records the current
        step's per-request ``(req_id, num_computed_tokens,
        num_scheduled_tokens)`` in batch order -- exactly the tensors the AR
        runner already passes to every model, so the model needs NO runner
        change.  The schedule lets ``forward`` /
        ``_adjust_positions_for_img2img`` gate per request instead of using
        the batch-wide padded length, and tells it where each chunk starts
        inside the request layout (split blocks) and how many logical
        positions precede this chunk (``num_computed``).
        """
        schedule: list[tuple[str, int, int]] = []
        for i, rid in enumerate(req_ids):
            n_computed = int(num_computed_tokens[i]) if i < len(num_computed_tokens) else 0
            n_scheduled = int(num_scheduled_tokens[i]) if i < len(num_scheduled_tokens) else 0
            schedule.append((str(rid), n_computed, n_scheduled))
        self._step_req_schedule = schedule

        if inputs_embeds is not None and input_ids is None and input_ids_buffer is not None:
            input_ids = input_ids_buffer
        return input_ids, positions
