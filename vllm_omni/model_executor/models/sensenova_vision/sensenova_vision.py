# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova-Vision-7B-MoT omni model.

SenseNova-Vision is a fork of Bagel with identical parameter-bearing modules.
This class reuses the MoT/ViT/VAE embedding logic from the BAGEL integration
and only overrides the SenseNovaVision checkpoint defaults plus additive features
(e.g. ``return_raw_latent``).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails

from vllm_omni.model_executor.models.bagel.bagel import (
    OmniBagelDummyInputsBuilder,
    OmniBagelForConditionalGeneration,
    OmniBagelMultiModalProcessor,
    OmniBagelProcessingInfo,
    OmniBagelProcessor,
)

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


def _fix_siglip_pos_encoding(embeddings) -> bool:
    """Bind a navit-exact ``interpolate_pos_encoding`` on a live SigLIP
    embeddings module.

    TWO bugs are repaired by shadowing the bound method:

    1. vLLM's ``SiglipVisionEmbeddings.interpolate_pos_encoding``
       (vllm/model_executor/models/siglip.py) reads
       ``self.position_embedding.weight.shape[1]`` -- the HIDDEN size --
       when reconstructing the 2-D positional grid; with the SigLIP-B/400
       table being (4900, 1152) any non-square ViT feed crashes with
       ``shape '[1, 33, 33, 1152]' invalid for input of size 5644800``
       (sqrt(1152) ~= 33).  Square feeds never reach that branch (the early
       return fires), which is why the fixed 980x980 base path worked.
    2. A corrected BICUBIC resample is STILL wrong for this checkpoint:
       upstream never interpolates the SigLIP position table.
       ``modeling/bagel/siglip_navit.py`` (the class the checkpoint ships)
       does ``patch_embeds + self.position_embedding(
       packed_flattened_position_ids)`` -- an EXACT-row lookup at ids
       ``h * 70 + w`` (``get_flattened_position_ids_extrapolate``), so any
       grid up to 70x70 lands entirely inside the trained 4900-row table,
       and interpolated/blended rows are OOD inputs the AR never saw
       (observed: out_13 degraded every img2img modality while
       understanding stayed healthy).

    The bound replacement therefore performs the navit-exact lookup:
    ``table[h * grid_side + w]`` for the fed grid (token order matches the
    conv row-major flattening == upstream ``patchify`` chpwq->hwpqc), with
    the original early-return retained for exact-square full-grid feeds
    (both branches produce identical rows there).  Idempotent; raises-
    free; returns False if the object does not look like SigLIP
    embeddings.
    """
    pos_embed = getattr(embeddings, "position_embedding", None)
    weight = getattr(pos_embed, "weight", None)
    if (
        weight is None
        or weight.ndim != 2
        or not hasattr(embeddings, "patch_size")
        or not torch.is_tensor(getattr(embeddings, "position_ids", None))
        or not callable(getattr(embeddings, "interpolate_pos_encoding", None))
    ):
        return False
    if getattr(embeddings, "_sensenova_pos_interp_fixed", False):
        return True
    grid = int(weight.shape[0] ** 0.5)
    if grid * grid != weight.shape[0]:
        # Not a square position table (never true for SigLIP); leave alone.
        return False

    import types as _types

    def _navit_pos_encoding(self, emb: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Exact-row lookup mirroring siglip_navit.SiglipVisionEmbeddings."""
        table = self.position_embedding.weight  # (num_positions, dim)
        num_positions = table.shape[0]
        grid_side = int(num_positions**0.5)
        gh, gw = height // self.patch_size, width // self.patch_size
        if emb.shape[1] == num_positions and height == width and gh * gw == num_positions:
            # Exact-square full-grid feed: direct lookup and the stock early
            # return yield the SAME rows; keep the cheap path.
            return self.position_embedding(self.position_ids)
        if gh > grid_side or gw > grid_side:
            raise RuntimeError(
                f"ViT grid {gh}x{gw} exceeds the learned position table "
                f"side {grid_side}; the checkpoint has no rows beyond it."
            )
        device = table.device
        h_coords = torch.arange(gh, device=device)
        w_coords = torch.arange(gw, device=device)
        pos_ids = (h_coords[:, None] * grid_side + w_coords).reshape(-1)
        return table[pos_ids].unsqueeze(0)  # (1, gh*gw, dim)

    embeddings._sensenova_pos_interp_fixed = True
    embeddings.interpolate_pos_encoding = _types.MethodType(_navit_pos_encoding, embeddings)
    return True


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


def _sensenova_vit_patch_count(img_h: int, img_w: int) -> int:
    """Aspect-aware ViT patch count for an input image at original resolution.

    Mirrors the model-side embedding chain (``_resize_to_stride`` then
    ``_sensenova_vit_resize_dims``) so the processor's ``image`` / ``img2img``
    placeholder token counts exactly equal the number of embedding rows the AR
    model produces (VAE resize first, then the ViT transform applied to the
    VAE-resized image, matching upstream ``InterleaveInferencer``).
    """
    vae_h, vae_w = _sensenova_vae_resize_dims(int(img_h), int(img_w))
    vit_h, vit_w = _sensenova_vit_resize_dims(vae_h, vae_w)
    return (vit_h // SENSENOVA_VISION_VIT_STRIDE) * (vit_w // SENSENOVA_VISION_VIT_STRIDE)


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
        is_img2img = kwargs.pop("is_img2img", False)
        if images is not None and not is_img2img:
            # Raw (aspect-preserving) pixels: no square pre-resize.  Mirror the
            # base img2img raw branch; the orchestrating ``_call_hf_processor``
            # keeps the image/img2img key names distinct.
            from vllm.transformers_utils.processors.bagel import BagelProcessorKwargs

            output_kwargs = self._merge_kwargs(
                BagelProcessorKwargs,
                tokenizer_init_kwargs=self.tokenizer.init_kwargs,
                **kwargs,
            )
            image_kwargs = dict(output_kwargs["images_kwargs"])
            image_kwargs["do_resize"] = False
            image_kwargs["do_rescale"] = True
            image_kwargs.setdefault("return_tensors", "pt")
            pixel_values = self.image_processor(images, **image_kwargs)

            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"]) if text is not None else None

            if pixel_values is not None and text_inputs is not None:
                combined = dict(text_inputs)
                combined["pixel_values"] = pixel_values["pixel_values"]
                return BatchFeature(combined)
            elif pixel_values is not None:
                return pixel_values
            elif text_inputs is not None:
                return BatchFeature(dict(text_inputs))
            else:
                return BatchFeature({})

        return super().__call__(text, images, is_img2img=is_img2img, **kwargs)


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

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs,
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptReplacement]:
        """Build prompt replacements with the SenseNova-Vision VAE transform.

        Mirrors :meth:`OmniBagelMultiModalProcessor._get_prompt_updates` but
        sizes the ``<|fim_middle|>`` VAE placeholder run with the official
        ``ImageTransform(1024, 512, 16)`` short-edge floor, matching the AR
        model's encoded latent grid (``_resize_to_stride`` override).  The two
        must agree or the placeholder token count will not equal the embedding
        length.  Otherwise identical to the base implementation.
        """
        replacements = super()._get_prompt_updates(mm_items, hf_processor_mm_kwargs, out_mm_kwargs)

        replacements = list(replacements)
        tokenizer = self.info.get_tokenizer()
        img2img_token_id = tokenizer.get_vocab().get("<|fim_middle|>")
        if img2img_token_id is None:
            return replacements
        image_token_id = tokenizer.get_vocab().get("<|image_pad|>")

        hf_config = self.info.get_hf_config()

        latent_patch_size = getattr(hf_config, "latent_patch_size", 2)
        downsample = hf_config.vae_config.get("downsample", 8)
        latent_downsample = downsample * latent_patch_size

        # Original input image size for a given item (width/height in original
        # pixel space, before any resize). Falls back to the ViT max square.
        def _img_size(item_idx: int, modality: str):
            item = mm_items.get_items(modality, (ImageProcessorItems, ImageEmbeddingItems))
            if hasattr(item, "get_image_size"):
                try:
                    size = item.get_image_size(item_idx)
                    return int(size.height), int(size.width)
                except Exception:
                    pass
            return SENSENOVA_VISION_VIT_MAX_SIZE, SENSENOVA_VISION_VIT_MAX_SIZE

        def get_image_replacement(item_idx: int):
            h, w = _img_size(item_idx, "image")
            return [image_token_id] * _sensenova_vit_patch_count(h, w)

        def get_img2img_replacement(item_idx: int):
            h, w = _img_size(item_idx, "img2img")

            # Two-stage official transform: VAE resize first, then the ViT
            # transform applied to the VAE-RESIZED image (upstream
            # interleave_inference L325 + update_context_image). Both stages
            # are aspect-preserving; the ViT patch count follows the aspect
            # ratio (capped at 4900 = 70x70, never exceeds the old square).
            new_h, new_w = _sensenova_vae_resize_dims(int(h), int(w))
            vit_h, vit_w = _sensenova_vit_resize_dims(new_h, new_w)
            num_vae_patches = (new_h // latent_downsample) * (new_w // latent_downsample)
            num_vit_patches = (vit_h // SENSENOVA_VISION_VIT_STRIDE) * (vit_w // SENSENOVA_VISION_VIT_STRIDE)
            # Upstream-exact layout (Bagel prepare_vae_images /
            # prepare_vit_images): each block bracketed by <|vision_start|> ...
            # <|vision_end|>, blocks ADJACENT - no <|fim_middle|> placeholder
            # run and no separator token ever appear in upstream sequences.
            #
            # EVERY slot in the expanded block carries an mm embedding,
            # mirroring upstream, which assigns embed_tokens(start/end_of_image)
            # to the marker rows and computed VAE-latent / ViT-patch embeddings
            # to the patch rows (_process_img2img_input builds exactly that
            # combined tensor). This is REQUIRED for vLLM's engine-side
            # placement: PlaceholderRange positions are matched to embedding
            # rows by the RUNNING COUNT of is_embed=True slots, so any False
            # slot interleaved INSIDE the placeholder shifts every subsequent
            # embedding onto the wrong token (observed GPU-wide corruption in
            # out_10 with marker rows marked False). all-True keeps the
            # position->row identity mapping exact.
            start_of_image_id = tokenizer.get_vocab()["<|vision_start|>"]
            end_of_image_id = tokenizer.get_vocab()["<|vision_end|>"]
            tokens = (
                [start_of_image_id]
                + [img2img_token_id] * num_vae_patches
                + [end_of_image_id]
                + [start_of_image_id]
                + [img2img_token_id] * num_vit_patches
                + [end_of_image_id]
            )

            return PromptUpdateDetails.from_seq(tokens)

        # Replace the image (img2text) and img2img placeholder updates with
        # aspect-aware resized versions; keep everything else the base produced.
        out: list[PromptReplacement] = []
        for r in replacements:
            if r.modality == "image":
                r = PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_image_replacement,
                )
            elif r.modality == "img2img":
                r = PromptReplacement(
                    modality="img2img",
                    target=[img2img_token_id],
                    replacement=get_img2img_replacement,
                )
            out.append(r)
        return out


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

    def _resize_to_stride(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Resize img2img pixel values to the official SenseNova-Vision VAE grid.

        Overrides the BAGEL base (whose short-edge floor is ``min(256, max)``)
        with the official ``ImageTransform(1024, 512, 16)``: the long edge is
        clamped to 1024 and the short edge pushed up to at least 512.  The
        base ``_process_img2img_input`` calls this per image, so the resulting
        ``image_shape`` cached in ``kv_metadata["image_shape"]`` lands the DiT
        output at the official resolution (e.g. a (375, 500) input becomes
        (368, 496) -> 512-aligned (512, 688)).
        """
        H, W = pixel_values.shape[2], pixel_values.shape[3]
        new_H, new_W = _sensenova_vae_resize_dims(H, W)
        if new_H != H or new_W != W:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=(new_H, new_W), mode="bicubic", align_corners=False
            )
        return pixel_values

    def _encode_vit_embeddings(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Encode img2img images through the ViT with upstream sizing.

        Upstream applies ``ImageTransform(980, 224, 14)`` -- an ASPECT-
        PRESERVING resize -- to the VAE-resized image before the ViT, so the
        patch count follows the aspect ratio instead of being a fixed square
        (the vLLM-core base resizes to ``image_size x image_size`` and emits
        exactly 70x70 patches).  This changes two things vs the base:

        - per-image ViT grids that follow aspect ratio (bicubic resize to the
          ``_sensenova_vit_resize_dims`` grid; must stay in lockstep with the
          processor's ``num_vit_patches`` placeholder count);
        - SigLIP position rows come from an EXACT table lookup (upstream
          siglip_navit never interpolates: ids are ``h * 70 + w`` into the
          trained 4900-row table, so any grid <= 70x70 is fully
          in-distribution).  vLLM's ``interpolate_pos_encoding`` both has a
          hidden-dim mixup (crashes on non-square feeds) and would bicubically
          blend rows -- OOD poison, see out_13 -- so
          :func:`_fix_siglip_pos_encoding` binds the navit-exact lookup.
        """
        embeddings_module = getattr(getattr(self.vit_model, "vision_model", None), "embeddings", None)
        if not _fix_siglip_pos_encoding(embeddings_module):
            raise RuntimeError(
                "Could not locate SiglipVisionEmbeddings on vit_model; "
                "aspect-preserving ViT grids require the pos-encoding fix."
            )
        vit_embeds = []
        for i in range(pixel_values.shape[0]):
            single_pv = pixel_values[i : i + 1]
            H, W = single_pv.shape[2:]
            vit_h, vit_w = _sensenova_vit_resize_dims(H, W)
            if (vit_h, vit_w) != (H, W):
                single_pv = torch.nn.functional.interpolate(
                    single_pv, size=(vit_h, vit_w), mode="bicubic", align_corners=False
                )
            features = self.vit_model(single_pv, interpolate_pos_encoding=True)
            embed = self.connector(features)
            num_patches = embed.shape[1]
            hidden = embed.shape[2]
            ph = self.config.vit_config.patch_size
            h_coords = torch.arange(vit_h // ph, device=embed.device)
            w_coords = torch.arange(vit_w // ph, device=embed.device)
            position_ids = (h_coords[:, None] * self.config.vit_max_num_patch_per_side + w_coords).flatten()
            position_ids = position_ids.unsqueeze(0).expand(1, -1).flatten()
            pos_embeds = self.vit_pos_embed(position_ids)
            pos_embeds = pos_embeds.reshape(1, num_patches, hidden)
            vit_embeds.append(embed + pos_embeds.to(embed.device))
        return tuple(vit_embeds)

    def _process_img2text_input(self, multimodal_input):
        """Base img2text (understanding) embedding, but with upstream sizing.

        The vLLM-core ``_process_image_input`` feeds the SigLIP a fixed
        ``image_size x image_size`` square (980x980 -> 70x70 = 4900 patches)
        and builds the pos-ids grid from ``image_size // patch_size``.  Upstream
        instead runs the VAE transform then the ViT transform on the ORIGINAL
        image, so the patch count follows the aspect ratio (no VAE latent
        encoding for understanding).  This mirrors the img2img path: VAE-resize
        per image, then :meth:`_encode_vit_embeddings` (which applies the ViT
        resize internally and the navit-exact SigLIP pos encoding).

        The returned per-image embeddings must contain exactly one row per
        ``<|image_pad|>`` placeholder token (``_sensenova_vit_patch_count``),
        matching the processor's aspect-aware placeholder sizing.
        """
        pixel_values = multimodal_input["pixel_values"]
        if pixel_values.ndim == 5:
            b, n, c, h, w = pixel_values.shape
            pixel_values = pixel_values.reshape(b * n, c, h, w)

        num_images = pixel_values.shape[0]
        if self._ropes_pending:
            self._ropes_pending.clear()

        vae_resized = [self._resize_to_stride(pixel_values[i : i + 1]) for i in range(num_images)]
        vit_embeddings = [emb for pv in vae_resized for emb in self._encode_vit_embeddings(pv)]
        # _encode_vit_embeddings yields (1, N, hidden) per image; the engine's
        # sanity_check_mm_encoder_outputs requires 2D (N, hidden) embeddings
        # (the same shape the img2img path returns), so drop the batch dim.
        return tuple(e.reshape(-1, e.shape[-1]) for e in vit_embeddings)

    def _process_img2img_input(self, multimodal_input):
        """Base img2img embedding, but ViT-encoded at upstream sizing.

        The vLLM-core ``_process_img2img_input`` bicubically squashes the ViT
        feed to a fixed 980x980 square (70x70 = 4900 patches regardless of
        aspect) and its core ``_process_image_input`` builds the pos-ids grid
        from ``image_size // patch_size``.  Upstream instead applies
        ``ImageTransform(980, 224, 14)`` to the VAE-resized image, so patch
        count follows aspect ratio.  This method replicates the base flow but
        swaps the ViT encoding for :meth:`_encode_vit_embeddings`.
        """
        pixel_values = multimodal_input["pixel_values"]
        if pixel_values.ndim == 5:
            b, n, c, h, w = pixel_values.shape
            pixel_values = pixel_values.reshape(b * n, c, h, w)

        num_images = pixel_values.shape[0]
        p = self.latent_patch_size
        timestep = 0

        if self._ropes_pending:
            self._ropes_pending.clear()

        # Upstream runs the ViT transform on the VAE-transformed image
        # (inferencer.update_context_image); do the same here.  _encode_
        # vit_embeddings handles its own ViT-grid sizing per image -- never
        # assume a fixed buffer shape equal to the raw input (profile-run
        # dummies disagree) nor a uniform aspect ratio across the batch.
        vae_resized = [self._resize_to_stride(pixel_values[i : i + 1]) for i in range(num_images)]
        vit_embeddings_tuple = tuple(emb for pv in vae_resized for emb in self._encode_vit_embeddings(pv))

        marker_ids = torch.tensor(
            [self._start_of_image_id, self._end_of_image_id],
            device=pixel_values.device,
            dtype=torch.long,
        )
        marker_embeds = self.language_model.model.embed_tokens(marker_ids)
        start_embed = marker_embeds[0:1]
        end_embed = marker_embeds[1:2]

        results = []
        for i in range(num_images):
            single_pv = pixel_values[i : i + 1]
            single_pv = self._resize_to_stride(single_pv)
            H, W = single_pv.shape[2:]

            padded_latent = self.vae.encode(single_pv)
            h = H // self.latent_downsample
            w = W // self.latent_downsample

            latent = padded_latent[0][:, : h * p, : w * p]
            latent = latent.reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)

            vae_position_ids = self.get_flattened_position_ids(
                H,
                W,
                self.latent_downsample,
                max_num_patches_per_side=self.max_latent_size,
            )
            pos_embed = self.latent_pos_embed([vae_position_ids])
            packed_timesteps = torch.tensor([timestep], device=padded_latent.device)
            with torch.amp.autocast(self.device.type, dtype=torch.bfloat16):
                timestep_embeds = self.time_embedder(packed_timesteps.to(padded_latent))
            vae_embeds = self.vae2llm(latent) + timestep_embeds + pos_embed

            vit_emb_full = vit_embeddings_tuple[i] if i < len(vit_embeddings_tuple) else vit_embeddings_tuple[0]
            # _encode_vit_embeddings yields (1, N, hidden); drop the batch dim.
            vit_emb = vit_emb_full.reshape(-1, vit_emb_full.shape[-1])

            se = start_embed.to(vae_embeds.dtype)
            ee = end_embed.to(vae_embeds.dtype)
            combined = torch.cat([se, vae_embeds, ee, se, vit_emb, ee], dim=0)
            results.append(combined)

            num_vae = h * w + 2  # +2 for start/end markers
            num_vit = vit_emb.shape[0] + 2
            info = (num_vae, num_vit, int(H), int(W))
            self._pending_img2img_info.append(info)
            self._last_img2img_info = info

        return tuple(results)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """SenseNova-Vision img2img bookkeeping (upstream-exact block layout).

        Mirrors ``OmniBagelForConditionalGeneration.forward`` but with the
        SEPARATOR-FREE block span (``num_vae + num_vit`` tokens, see
        ``_get_prompt_updates`` above): the BAGEL base assumes a legacy
        ``<|fim_middle|>`` separator between the VAE and ViT sections that
        upstream sequences never contain.

        Also rewrites the understanding (img2text / dense_detection) path with
        the upstream-exact collapsed image-block scheme via
        :meth:`_adjust_positions_for_understanding`; text-only requests and
        ``<|fim_middle|>``-based img2img blocks are left untouched.
        """
        use_mot = False
        seq_len = inputs_embeds.shape[0] if inputs_embeds is not None else positions.shape[0]

        if self._pending_img2img_info:
            self._log_prompt_token_probe(input_ids)
            positions = self._adjust_positions_for_img2img(positions, input_ids)
            use_mot = True
        elif self._last_img2img_info is not None:
            info = self._last_img2img_info
            num_vae, num_vit, _, _ = info
            num_img2img = num_vae + num_vit  # no separator (upstream-exact)

            if seq_len >= num_img2img:
                self._pending_img2img_info = [info]
                positions = self._adjust_positions_for_img2img(positions, input_ids)
                use_mot = True
            else:
                rope = positions[seq_len - 1] + 1
                self._ropes_pending.append({"ropes": [rope]})

        if use_mot:
            return self._mot_forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

        positions = self._adjust_positions_for_understanding(positions, input_ids)
        # Text-only / img2text path: bypass the BAGEL base's img2img
        # bookkeeping (its separator-based span math does not apply here).
        return super(OmniBagelForConditionalGeneration, self).forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

    def _adjust_positions_for_understanding(
        self,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rewrite position IDs for the understanding path (upstream-exact).

        Upstream (``Bagel.prepare_vit_images``) collapses the ENTIRE image
        block -- ``[<|vision_start|>] patches [<|vision_end|>]`` -- to ONE
        shared logical position: ``packed_position_ids.extend(
        [curr_position_id] * (num_img_tokens + 2))``, text resumes at
        ``curr_position_id + 1``, and decode starts at the post-block rope.

        Our understanding (``image`` modality) prompts carry each image as a
        plain run of ``N`` ``<|image_pad|>`` tokens (no vision markers on this
        path), so the translation is: every ``<|image_pad|>`` RUN is one
        collapsed block.  For one block at the front (the standard img2text /
        dense_detection layout):

            leading text  -> 0 .. M-1          (unchanged)
            image block   -> M        (all N share)
            post text     -> M+1 ..          (threaded below)
            decode step k -> rope + k, rope = M + 1 + num_post_text

        Multi-image understanding requests (k blocks) each collapse to their
        own ``M_k`` (advancing by 1 per block plus interleaved text), matching
        upstream's per-image ``curr_position_id`` advance of exactly 1.

        Text-only requests and ``<|fim_middle|>``-based img2img requests
        contain no ``<|image_pad|>`` runs and are left untouched.  Decode
        continuity falls out of the exported rope: the prefill's last logical
        position is ``rope - 1``, so vLLM's vanilla ``L, L+1, ...`` decode
        positions equal ``rope, rope+1, ...`` when the same constant collapse
        is applied on decode steps.

        Slot mapping is untouched (computed by the runner from vanilla
        token-index positions before forward): this is a RoPE-only rewrite,
        exactly like :meth:`_adjust_positions_for_img2img`.
        """
        if input_ids is None or positions.numel() == 0:
            return positions

        # Split the concatenated batch per request using the same positions
        # reset-to-0 boundary detection as _adjust_positions_for_img2img.
        boundaries = [0]
        pos_list = positions.tolist()
        for i in range(1, len(pos_list)):
            if pos_list[i] < pos_list[i - 1]:
                boundaries.append(i)
        boundaries.append(len(pos_list))

        has_blocks = False
        new_positions = positions.clone()
        ids_list = input_ids.tolist()
        img2text_token = self._img2text_token_id

        for req_idx in range(len(boundaries) - 1):
            start = boundaries[req_idx]
            end = boundaries[req_idx + 1]
            req_ids = ids_list[start:end]

            # Locate the <|image_pad|> runs (each run = one image block).
            spans: list[tuple[int, int]] = []
            scan = 0
            while scan < len(req_ids):
                if req_ids[scan] == img2text_token:
                    run_start = scan
                    while scan < len(req_ids) and req_ids[scan] == img2text_token:
                        scan += 1
                    spans.append((run_start, scan - run_start))
                else:
                    scan += 1

            if not spans:
                continue
            has_blocks = True

            # Rebase logical positions per request.  Leading text keeps 0..M-1
            # (it already is, since vanilla prefill is sequential from 0).
            # Each block collapses to ONE shared position M_k; text after a
            # block resumes at the threaded cursor, so a request with k blocks
            # occupies ``#text_tokens + k`` logical positions.
            logical_m = 0
            block_end = 0
            for off, length in spans:
                # Text before this block (or between blocks) resumes at the
                # threaded cursor; gaps are measured in TOKEN offsets, while
                # the cursor advances one logical position per block.
                if off > block_end:
                    gap_len = off - block_end
                    new_positions[start + block_end : start + off] = torch.arange(
                        logical_m,
                        logical_m + gap_len,
                        device=positions.device,
                        dtype=positions.dtype,
                    )
                    logical_m += gap_len
                # The block collapses to ONE shared logical position.
                new_positions[start + off : start + off + length] = logical_m
                logical_m += 1
                block_end = off + length

            # Trailing text after the last block resumes from the cursor.
            # block_end is the LAST block's end offset within THIS request
            # (leading-gap requests start with block_end=0 and may have trailing
            # text after the final collapsed block).
            trailing_start = block_end
            trailing_len = len(req_ids) - trailing_start
            if trailing_len > 0:
                new_positions[start + trailing_start : end] = torch.arange(
                    logical_m,
                    logical_m + trailing_len,
                    device=positions.device,
                    dtype=positions.dtype,
                )
            logical_m += trailing_len

            # Export the collapsed post-prefill rope (upstream ``new_rope``):
            # decode continues at logical_m == M_k + 1 + trailing_text_len.
            # prefill_position_count lets get_kv_transfer_metadata compute
            # decode-step rope offsets without relying on num_computed_tokens
            # being a prompt length.
            rope = logical_m
            self._ropes_pending.append(
                {
                    "ropes": [rope],
                    "prefill_position_count": len(req_ids),
                }
            )

        if not has_blocks:
            return positions
        return new_positions

    def _adjust_positions_for_img2img(
        self,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rewrite position IDs for img2img (upstream-exact block layout).

        Blocks are ``[<|vision_start|>] fim-patches [<|vision_end|>]
        [<|vision_start|>] fim-patches [<|vision_end|>]`` -- ADJACENT, no
        separator token (see ``_get_prompt_updates``).  Detected via the
        leading ``<|vision_start|>`` (thinking-mode pre-text never carries
        vision tokens):

            pre_text -> 0 .. M-1
            VAE sect -> M       (all share, markers included)
            ViT sect -> M+1     (all share, markers included)
            post_text-> M+2, M+3, ...

        When M=0 (standard img2img) this reduces to VAE->0, ViT->1, text->2..

        A single request may carry SEVERAL img2img blocks (multi-image
        requests): every block collapses to two logical positions anchored at
        its own M, with interleaved text resuming at M+2, so a request with
        k blocks occupies ``#text_tokens + 2*k`` logical positions.
        """
        info_list = self._pending_img2img_info
        self._pending_img2img_info = []

        if not info_list:
            self._vae_token_mask = None
            self._has_vae_tokens = False
            self._has_non_vae_tokens = True
            return positions

        boundaries = [0]
        # Copy positions to the host once: indexing the CUDA tensor element by
        # element in the loop below would sync the device on every iteration.
        pos_list = positions.tolist()
        for i in range(1, len(pos_list)):
            if pos_list[i] < pos_list[i - 1]:
                boundaries.append(i)
        boundaries.append(len(pos_list))

        num_requests = len(boundaries) - 1
        new_positions = positions.clone()
        vae_mask = torch.zeros(len(positions), dtype=torch.bool, device=positions.device)

        img2img_idx = 0
        # Host copy of input_ids for placeholder matching (same rationale as
        # the positions copy above: per-element device indexing would sync on
        # every token).
        ids_list = input_ids.tolist() if input_ids is not None else None
        for req_idx in range(num_requests):
            start = boundaries[req_idx]
            end = boundaries[req_idx + 1]
            req_len = end - start

            # Match this request's img2img blocks against the pending infos.
            # A block is ``[<|vision_start|>] fim-patches [<|vision_end|>]
            # [<|vision_start|>] fim-patches [<|vision_end|>]`` -- exactly
            # ``num_vae + num_vit`` tokens, NO separator -- so adjacent blocks
            # concatenate into longer runs that this scan still splits
            # correctly by advancing block_len tokens per matched info.
            # Infos left over stay queued for the following requests in the
            # batch.
            spans = []
            if ids_list is not None and img2img_idx < len(info_list):
                req_ids = ids_list[start:end]
                soi = self._start_of_image_id
                eoi = self._end_of_image_id
                fim = self._img2img_token_id
                scan = 0
                info_i = img2img_idx
                while info_i < len(info_list):
                    num_vae, num_vit = info_list[info_i][0], info_list[info_i][1]
                    block_len = num_vae + num_vit  # no separator
                    while scan < req_len and req_ids[scan] != soi:
                        scan += 1
                    if scan >= req_len or req_len - scan < block_len:
                        break
                    blk = req_ids[scan : scan + block_len]
                    if (
                        blk[0] != soi
                        or blk[num_vae - 1] != eoi
                        or blk[num_vae] != soi
                        or blk[-1] != eoi
                        or any(t != fim for t in blk[1 : num_vae - 1])
                        or any(t != fim for t in blk[num_vae + 1 : -1])
                    ):
                        break
                    spans.append((scan, *info_list[info_i]))
                    scan += block_len
                    info_i += 1

            if spans:
                # Logical positions are rebased per request: leading text keeps
                # 0..M1-1, every block collapses to TWO shared logical positions
                # (VAE section -> M, ViT section -> M+1) regardless of token
                # count, and text after a block resumes at M+2. NOTE: a block's
                # logical anchor M is NOT its token offset once earlier blocks
                # have compressed their tokens, hence the threaded cursor.
                first_off = spans[0][0]
                if first_off > 0:
                    new_positions[start : start + first_off] = torch.arange(
                        0, first_off, device=positions.device, dtype=positions.dtype
                    )
                logical_m = first_off
                for k, (off, num_vae, num_vit, img_H, img_W) in enumerate(spans):
                    img_start = start + off
                    vit_start = img_start + num_vae  # no separator
                    new_positions[img_start:vit_start] = logical_m  # VAE section (markers incl.)
                    new_positions[vit_start : vit_start + num_vit] = logical_m + 1  # ViT section
                    vae_lo = img_start + 1
                    vae_hi = img_start + num_vae - 1
                    if vae_hi > vae_lo:
                        vae_mask[vae_lo:vae_hi] = True
                    block_end = off + num_vae + num_vit
                    next_off = spans[k + 1][0] if k + 1 < len(spans) else req_len
                    gap_len = next_off - block_end
                    if gap_len > 0:
                        new_positions[start + block_end : start + block_end + gap_len] = torch.arange(
                            logical_m + 2,
                            logical_m + 2 + gap_len,
                            device=positions.device,
                            dtype=positions.dtype,
                        )
                    logical_m += 2 + gap_len
                self._ropes_pending.append(
                    {
                        "ropes": [logical_m],
                        "image_shape": [spans[-1][3], spans[-1][4]],
                        "prefill_position_count": req_len,
                    }
                )
                img2img_idx += len(spans)
                continue

            if img2img_idx < len(info_list):
                cur_info = info_list[img2img_idx]
            elif self._last_img2img_info is not None:
                cur_info = self._last_img2img_info
            else:
                cur_info = None

            if cur_info is not None:
                num_vae, num_vit, img_H, img_W = cur_info
                num_img2img = num_vae + num_vit  # no separator (upstream-exact)

                if req_len >= num_img2img:
                    pre_text_len = 0
                    if input_ids is not None:
                        req_ids_slice = input_ids[start:end]
                        indices = (req_ids_slice == self._start_of_image_id).nonzero(as_tuple=True)[0]
                        if indices.numel() > 0:
                            pre_text_len = int(indices[0].item())

                    M = pre_text_len
                    img_start = start + M
                    post_text_start = img_start + num_img2img

                    if M > 0:
                        new_positions[start:img_start] = torch.arange(
                            0, M, device=positions.device, dtype=positions.dtype
                        )

                    new_positions[img_start : img_start + num_vae] = M
                    vit_start = img_start + num_vae  # no separator
                    new_positions[vit_start : vit_start + num_vit] = M + 1

                    num_post_text = end - post_text_start
                    if num_post_text > 0:
                        new_positions[post_text_start:end] = torch.arange(
                            M + 2,
                            M + 2 + num_post_text,
                            device=positions.device,
                            dtype=positions.dtype,
                        )

                    vae_patches_start = img_start + 1
                    vae_patches_end = img_start + num_vae - 1
                    if vae_patches_end > vae_patches_start:
                        vae_mask[vae_patches_start:vae_patches_end] = True

                    rope = M + 2 + num_post_text
                    self._ropes_pending.append(
                        {
                            "ropes": [rope],
                            "image_shape": [img_H, img_W],
                            "prefill_position_count": req_len,
                        }
                    )
                    img2img_idx += 1
                    continue

            rope = int(new_positions[end - 1].item()) + 1
            self._ropes_pending.append({"ropes": [rope]})

        # Resolve mask occupancy once here (the only .any() syncs on this path)
        # and cache it; the per-layer routing reads these flags instead of
        # re-checking the mask on every decoder layer.
        has_vae = bool(vae_mask.any())
        self._vae_token_mask = vae_mask if has_vae else None
        self._has_vae_tokens = has_vae
        self._has_non_vae_tokens = bool((~vae_mask).any()) if has_vae else True
        return new_positions
