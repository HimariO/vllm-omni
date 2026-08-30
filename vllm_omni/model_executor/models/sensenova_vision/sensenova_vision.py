# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova-Vision-7B-MoT omni model.

SenseNova-Vision is a fork of Bagel with identical parameter-bearing modules.
This class reuses the MoT/ViT/VAE embedding logic from the BAGEL integration
and only overrides the SenseNovaVision checkpoint defaults plus additive features
(e.g. ``return_raw_latent``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from transformers import BatchFeature
from vllm.config import VllmConfig
from vllm.logger import init_logger
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

        Standalone (never calls the BAGEL ``super()`` implementation so the
        legacy separator/`+1` block layout can never leak back in).  Each
        modality contributes ONE ``PromptReplacement`` whose ``target`` is a
        SINGLE placeholder token and whose ``replacement`` is a callable that
        expands ONE placeholder into the full per-image block (one block per
        placeholder token in the prompt, in order).  ``_bind_and_group_updates``
        resolves the single update once per item index, and
        ``apply_token_matches``/``_iter_placeholders`` match the single-token
        targets sequentially - so N identical placeholders yield N identical
        blocks with unambiguous binding.  Each block's full content is
        ``[<|vision_start|>] fim-patches [<|vision_end|>]
        [<|vision_start|>] fim-patches [<|vision_end|>]`` for img2img, or a
        bare run of ``<|image_pad|>`` for understanding, with ``is_embed=None``
        (every slot embedded) matching the AR model's embedding assembly.
        """
        tokenizer = self.info.get_tokenizer()
        vocab = tokenizer.get_vocab()

        image_token_id = vocab.get("<|image_pad|>")
        img2img_token_id = vocab.get("<|fim_middle|>")
        start_of_image_id = vocab.get("<|vision_start|>")
        end_of_image_id = vocab.get("<|vision_end|>")

        hf_config = self.info.get_hf_config()
        latent_patch_size = getattr(hf_config, "latent_patch_size", 2)
        downsample = hf_config.vae_config.get("downsample", 8)
        latent_downsample = downsample * latent_patch_size

        # Original input HxW for a given item (before any resize).  Falls back
        # to the ViT max square when the size cannot be read.
        def _img_size(item_idx: int, modality: str):
            item = mm_items.get_items(modality, (ImageProcessorItems, ImageEmbeddingItems))
            if hasattr(item, "get_image_size"):
                try:
                    size = item.get_image_size(item_idx)
                    return int(size.height), int(size.width)
                except Exception:
                    pass
            return SENSENOVA_VISION_VIT_MAX_SIZE, SENSENOVA_VISION_VIT_MAX_SIZE

        def understanding_block(item_idx: int) -> PromptUpdateDetails:
            # Aspect-aware understanding placeholder run:
            # ``_sensenova_understanding_patch_count`` applies the same
            # VAE->ViT two-stage sizing as the model's img2text path
            # (``_process_img2text_input``).
            h, w = _img_size(item_idx, "image")
            return PromptUpdateDetails.from_seq([image_token_id] * _sensenova_understanding_patch_count(h, w))

        def img2img_block(item_idx: int) -> PromptUpdateDetails:
            h, w = _img_size(item_idx, "img2img")

            # Two-stage official transform: VAE resize first, then the ViT
            # transform applied to the VAE-RESIZED image (upstream
            # interleave_inference L325 + update_context_image). Both stages
            # are aspect-preserving; the ViT patch count follows the aspect
            # ratio (capped at 4900 = 70x70, never exceeds the old square).
            new_h, new_w = _sensenova_vae_resize_dims(int(h), int(w))
            num_vae_patches = (new_h // latent_downsample) * (new_w // latent_downsample)
            num_vit_patches = _sensenova_vit_patch_count(new_h, new_w)
            # Upstream-exact layout (Bagel prepare_vae_images /
            # prepare_vit_images): each block bracketed by <|vision_start|> ...
            # <|vision_end|>, blocks ADJACENT - no <|fim_middle|> placeholder
            # run and no separator token ever appear in upstream sequences.
            #
            # EVERY slot in the expanded block carries an mm embedding
            # (``is_embed=None``), mirroring upstream, which assigns
            # embed_tokens(start/end_of_image) to the marker rows and computed
            # VAE-latent / ViT-patch embeddings to the patch rows
            # (``_process_img2img_input`` builds exactly that combined tensor).
            # This is REQUIRED for vLLM's engine-side placement:
            # PlaceholderRange positions are matched to embedding rows by the
            # RUNNING COUNT of is_embed=True slots, so any False slot
            # interleaved INSIDE the placeholder shifts every subsequent
            # embedding onto the wrong token (observed GPU-wide corruption in
            # out_10 with marker rows marked False). all-True keeps the
            # position->row identity mapping exact.
            tokens = (
                [start_of_image_id]
                + [img2img_token_id] * num_vae_patches
                + [end_of_image_id]
                + [start_of_image_id]
                + [img2img_token_id] * num_vit_patches
                + [end_of_image_id]
            )
            return PromptUpdateDetails.from_seq(tokens)

        out: list[PromptReplacement] = []
        if image_token_id is not None and "image" in mm_items.get_all_counts():
            out.append(
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=understanding_block,
                )
            )
        if img2img_token_id is not None and start_of_image_id is not None and end_of_image_id is not None:
            if "img2img" in mm_items.get_all_counts():
                out.append(
                    PromptReplacement(
                        modality="img2img",
                        target=[img2img_token_id],
                        replacement=img2img_block,
                    )
                )
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
            # The ViT grid for the position IDs must be the EXACT patch grid
            # produced by the resize (and equal to the placeholder patch count).
            # ``vit_h // ph`` is correct only when ``vit_h`` is patch-aligned;
            # use the same count helper the processor uses (never assume the
            # resized dims divide evenly by the patch size).
            num_patches_h = vit_h // ph
            num_patches_w = vit_w // ph
            h_coords = torch.arange(num_patches_h, device=embed.device)
            w_coords = torch.arange(num_patches_w, device=embed.device)
            position_ids = (h_coords[:, None] * self.config.vit_max_num_patch_per_side + w_coords).flatten()
            position_ids = position_ids.unsqueeze(0).expand(1, -1).flatten()
            pos_embeds = self.vit_pos_embed(position_ids)
            pos_embeds = pos_embeds.reshape(1, num_patches, hidden)
            vit_embeds.append(embed + pos_embeds.to(embed.device))
        return tuple(vit_embeds)

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
        upstream sequences never contain.  Text-only / img2text requests
        bypass this path entirely and fall through to the base forward.

        Gating is per-request, from the step schedule captured by
        ``prepare_runner_inputs`` (``_step_req_schedule``): the batch-wide
        ``inputs_embeds.shape[0]`` / ``positions.shape[0]`` length is replaced
        by each request's ``(num_computed, num_scheduled)`` so a padded
        CUDA-graph batch or a sibling request can never change how a
        request's chunk is classified.  A request enters the img2img MoT path
        when this step contains an active layout for it (``_img2img_layouts``)
        or when pending geometry is queued.  See ``_adjust_positions_for_img2img``
        for the per-request collapse / partial-collapse / rope-only logic.
        """
        # _adjust_positions_for_img2img consumes _step_req_schedule; do not
        # clear it here.
        schedule = self._step_req_schedule or []

        use_mot = False
        any_img2img = bool(self._pending_img2img_info) or bool(self._img2img_layouts)
        if not any_img2img:
            for rid, _n_computed, _n_scheduled in schedule:
                if rid in self._img2img_layouts:
                    any_img2img = True
                    break

        if any_img2img:
            self._log_prompt_token_probe(input_ids)
            positions = self._adjust_positions_for_img2img(positions, input_ids)
            use_mot = True

        if use_mot:
            return self._mot_forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

        # Text-only / img2text path: bypass the BAGEL base's img2img
        # bookkeeping (its separator-based span math does not apply here).
        return super(OmniBagelForConditionalGeneration, self).forward(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

    def _adjust_positions_for_img2img(
        self,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rewrite position IDs for img2img (upstream-exact block layout).

        Blocks are ``[<|vision_start|>] fim-patches [<|vision_end|>]
        [<|vision_start|>] fim-patches [<|vision_end|>]`` -- ADJACENT, no
        separator token (see ``_get_prompt_updates``):

            pre_text -> 0 .. M-1
            VAE sect -> M       (all share)
            ViT sect -> M+1     (all share)
            post_text-> M+2, M+3, ...

        When M=0 (standard img2img) this reduces to VAE->0, ViT->1, text->2..

        A single request may carry SEVERAL img2img blocks (multi-image
        requests): every block collapses to two logical positions anchored at
        its own M, with interleaved text resuming at M+2, so a request with
        k blocks occupies ``#text_tokens + 2*k`` logical positions.

        The rewrite is driven by the per-request step schedule captured by
        ``prepare_runner_inputs`` (``_step_req_schedule``) instead of the
        batch-wide ``positions``/``inputs_embeds`` length and the
        position-reset boundary detection (both unreliable for a padded
        CUDA-graph batch and for continuation prefill chunks where positions
        CONTINUE from each request's ``num_computed_tokens``).  Per-request
        layout state in ``_img2img_layouts`` survives across forward() calls
        so blocks split by chunked prefill collapse incrementally without
        drift:

        - first chunk: geometry is bound positionally from
          ``_pending_img2img_info``.  If the window ends mid-block the
          partial block's geometry is bound too and the phase machine
          resumes it in the continuation chunk.
        - continuation chunk: the layout's phase/anchors decide collapse.
          The encoder re-encodes a block that spans a chunk boundary, so the
          FIFO is refilled with a STALE re-encode of that block; the stale
          entry is skipped (once) at geometry resolution ONLY when the chunk
          started mid-block (``_stale_possible``), never for a chunk whose
          blocks are fully contained.
        - first decode step: prefill has ended exactly when a step is a
          single token (``num_scheduled == 1``), the layout is idle in text
          phase, and the token is not a block SOI.  ``num_computed ==
          prompt_len`` there, so the FULL rope metadata (``image_shape`` +
          ``prefill_position_count``) is emitted on that step (the
          ``flush_pending_metadata`` guard keeps it across later plain
          decode ropes) and the layout is pruned.  The scheduler processes
          one token per request on every decode step, and chunked prefill
          never reaches prompt-end with a >1-token step misclassified: only
          a 1-token prefill chunk that happens to be pure text could
          miss-fire, and the rope arithmetic stays self-consistent there
          (prefill_rope and prefill_position_count are both captured before
          that token).

        One ropes-pending entry is appended PER SEGMENT in batch order
        (1:1 mapping with ``flush_pending_metadata``); text-only siblings get
        the plain rope fallback, exactly like the BAGEL base.
        """
        info_list = self._pending_img2img_info
        self._pending_img2img_info = []

        schedule = self._step_req_schedule
        self._step_req_schedule = []
        if not schedule:
            # No schedule (legacy/warmup direct call): treat the whole batch
            # as one segment and hand out the pending infos in order.
            schedule = [("", 0, len(positions))]
        if len(schedule) == 1:
            # Single-request schedule may describe only the unpadded length;
            # clamp to the actual positions length.
            rid, n_computed, n_scheduled = schedule[0]
            schedule = [(str(rid), n_computed, min(n_scheduled, len(positions)))]

        # Per-request token segments in batch order.  ``(rid, start, end,
        # n_computed, n_scheduled)`` where [start, end) is the token slice of
        # this request inside the tensor.
        req_segments = []
        token_off = 0
        for rid, n_computed, n_scheduled in schedule:
            end = min(token_off + max(n_scheduled, 0), len(positions))
            req_segments.append((str(rid), token_off, end, n_computed, n_scheduled))
            token_off += max(n_scheduled, 0)

        new_positions = positions.clone()
        vae_mask = torch.zeros(len(positions), dtype=torch.bool, device=positions.device)
        # Host copy of input_ids for placeholder matching (same rationale as
        # the positions copy above: per-element device indexing would sync on
        # every iteration).
        ids_list = input_ids.tolist() if input_ids is not None else None

        info_holder = [0]  # next unconsumed entry of THIS chunk's info_list
        any_active = False
        for rid, start, end, n_computed, n_scheduled in req_segments:
            req_len = end - start
            if req_len <= 0:
                continue

            layout = self._img2img_layouts.get(rid)
            if layout is None:
                # First chunk: bind geometry positionally from the FIFO,
                # including a PARTIAL block cut by the window end.
                spans, partial, bound = self._match_leading_blocks(ids_list, start, end, info_list, info_holder)
                if bound:
                    layout = self._enter_layout_from_span(rid, spans, partial, n_computed)
                    self._img2img_layouts[rid] = layout

            if layout is None:
                # text-only sibling / decode-after-prune: plain rope
                # fallback (positions untouched).
                rope = int(new_positions[end - 1].item()) + 1 if end > start else 0
                self._ropes_pending.append({"ropes": [rope]})
                continue

            # First decode step = prefill done: exactly one token, layout
            # idle in text phase, no block SOI in the chunk.  At that point
            # ``num_computed == prompt_len``, so the full rope metadata is
            # captured (the flush_pending_metadata guard keeps it across the
            # later plain decode ropes) and the layout is pruned.
            is_first_decode = False
            if (
                n_scheduled == 1
                and req_len == 1
                and layout.get("phase") == "text"
                and int(layout.get("_remaining", 0)) <= 0
            ):
                tok = ids_list[start] if ids_list is not None else None
                if tok is None or tok != self._start_of_image_id:
                    is_first_decode = True

            if is_first_decode:
                self._emit_layout_rope(layout, n_computed, n_scheduled, done=True)
                self._img2img_layouts.pop(rid, None)
                continue

            any_active = True

            # Collapse the tokens in [start, end) according to the layout.
            self._collapse_chunk_into_layout(
                layout, start, end, new_positions, vae_mask, ids_list, info_list, info_holder
            )
            self._emit_layout_rope(layout, n_computed, n_scheduled, done=False)

        if not any_active:
            self._vae_token_mask = None
            self._has_vae_tokens = False
            self._has_non_vae_tokens = True
            return new_positions

        # Resolve mask occupancy once here (the only .any() syncs on this
        # path) and cache it; the per-layer routing reads these flags instead
        # of re-checking the mask on every decoder layer.
        has_vae = bool(vae_mask.any())
        self._vae_token_mask = vae_mask if has_vae else None
        self._has_vae_tokens = has_vae
        self._has_non_vae_tokens = bool((~vae_mask).any()) if has_vae else True
        return new_positions

    # ------------------------------------------------------------------
    # Per-request img2img layout helpers
    # ------------------------------------------------------------------

    def _match_leading_blocks(
        self,
        ids_list: list[int] | None,
        start: int,
        end: int,
        info_list: list[tuple[int, int, int, int]],
        info_holder: list[int],
    ) -> tuple[list[tuple[int, int, int, int, int]], tuple | None, bool]:
        """First-chunk binding against the pending infos, in order.

        Returns ``(spans, partial, bound)``:

        - ``spans``: complete blocks matched and consumed from the FIFO
          (``(off, num_vae, num_vit, H, W)``, offsets absolute in the batch).
        - ``partial``: ``(off, num_vae, num_vit, H, W)`` when the window ends
          mid-block -- its geometry is bound positionally (Bug C) and the
          phase machine resumes it in the continuation chunk.
        - ``bound``: True iff any geometry was bound.
        """
        spans = []
        partial = None
        if ids_list is None:
            return spans, partial, False
        section = ids_list[start:end]
        soi = self._start_of_image_id
        eoi = self._end_of_image_id
        fim = self._img2img_token_id
        scan = 0
        while True:
            while scan < len(section) and section[scan] != soi:
                scan += 1
            if scan >= len(section):
                break
            idx = info_holder[0]
            if idx >= len(info_list):
                break
            num_vae, num_vit = int(info_list[idx][0]), int(info_list[idx][1])
            block_len = num_vae + num_vit  # no separator
            if len(section) - scan < block_len:
                # Block starts but the window cuts it short: bind geometry.
                partial = (start + scan, num_vae, num_vit, int(info_list[idx][2]), int(info_list[idx][3]))
                info_holder[0] = idx + 1
                break
            blk = section[scan : scan + block_len]
            if not (
                blk[0] == soi
                and blk[num_vae - 1] == eoi
                and blk[num_vae] == soi
                and blk[-1] == eoi
                and all(t == fim for t in blk[1 : num_vae - 1])
                and all(t == fim for t in blk[num_vae + 1 : -1])
            ):
                break
            spans.append((start + scan, num_vae, num_vit, int(info_list[idx][2]), int(info_list[idx][3])))
            info_holder[0] = idx + 1
            scan += block_len
        return spans, partial, bool(spans) or partial is not None

    def _enter_layout_from_span(
        self,
        rid: str,
        spans: list[tuple[int, int, int, int, int]],
        partial: tuple | None,
        segment_base: int,
    ) -> dict[str, object]:
        """Initialize a fresh per-request layout from the matched geometry.

        The layout keeps a persistent logical cursor (``next_logical`` /
        ``next_text``) seeded from the raw position base (0 for a fresh
        request, ``num_computed_tokens`` for a continuation chunk whose
        layout was never bound), plus the image shape of the LAST block seen
        so far (``_last_img_shape``) so the final prefill chunk's rope
        metadata carries the correct ``image_shape``.
        """
        if partial is not None:
            last_shape = (int(partial[3]), int(partial[4]))
        else:
            last_shape = (int(spans[-1][3]), int(spans[-1][4]))
        layout = {
            "rid": rid,
            "phase": "text",
            "next_text": int(segment_base),
            "next_logical": int(segment_base),
            "_seeded_pos": int(segment_base),
            "_spans": spans,
            "_span_i": 0,
            "_block_info": None,
            "_last_img_shape": last_shape,
            "_last_completed_geometry": None,
            "_stale_possible": False,
            "_has_vae": False,
        }
        if partial is not None:
            layout["_block_info"] = (int(partial[1]), int(partial[2]), int(partial[3]), int(partial[4]))
        return layout

    def _resolve_block_geometry(
        self,
        layout: dict[str, object],
        info_list: list[tuple[int, int, int, int]],
        info_holder: list[int],
        abs_off: int,
    ) -> tuple[int | None, int | None]:
        """Geometry for the block whose SOI sits at absolute ``abs_off``.

        Resolution order: (1) the next matched span, (2) a seeded
        ``_block_info`` (partial first block), (3) the per-chunk FIFO --
        skipping ONE stale re-encode of the just-completed block, but ONLY
        when this chunk started mid-block (``_stale_possible``; the encoder
        re-encodes a boundary-spanning block, so its info is refilled at the
        head while the continuation chunk finishes it -- a fresh chunk whose
        blocks are fully contained has no stale entry and must not skip),
        (4) the last-completed geometry as a final fallback.
        """
        spans = layout.get("_spans") or []
        span_i = int(layout.get("_span_i", 0))
        if span_i < len(spans) and spans[span_i][0] == abs_off:
            sp = spans[span_i]
            layout["_span_i"] = span_i + 1
            layout["_last_img_shape"] = (int(sp[3]), int(sp[4]))
            return int(sp[1]), int(sp[2])

        block_info = layout.get("_block_info")
        if block_info is not None:
            layout["_block_info"] = None
            layout["_last_img_shape"] = (int(block_info[2]), int(block_info[3]))
            return int(block_info[0]), int(block_info[1])

        idx = info_holder[0]
        last = layout.get("_last_completed_geometry")
        if (
            layout.get("_stale_possible")
            and last is not None
            and idx < len(info_list)
            and tuple(info_list[idx][:2]) == tuple(last)
        ):
            idx += 1  # stale re-encode of the just-completed block
            layout["_stale_possible"] = False
        if idx < len(info_list):
            info = info_list[idx]
            info_holder[0] = idx + 1
            layout["_last_img_shape"] = (int(info[2]), int(info[3]))
            return int(info[0]), int(info[1])
        if last is not None:
            return int(last[0]), int(last[1])
        return None, None

    def _collapse_chunk_into_layout(
        self,
        layout: dict[str, object],
        start: int,
        end: int,
        new_positions: torch.Tensor,
        vae_mask: torch.Tensor,
        ids_list: list[int] | None,
        info_list: list[tuple[int, int, int, int]],
        info_holder: list[int],
    ) -> None:
        """Rewrite the tokens of one chunk into the per-request layout.

        The layout is a per-request phase machine over logical positions:

        - ``phase == "text"``: tokens get sequential logical positions from
          ``next_text`` until the next SOI opens a block.
        - ``phase == "vae"`` / ``"vit"``: the present section's tokens all
          share the section anchor; VAE patch rows (the <|fim_middle|>
          interior, excluding the SOI/EOI markers) are marked in
          ``vae_mask``.
        - when a block completes the layout's ``next_logical`` advances by 2
          and the following text resumes sequentially; the completion /
          prefill-done decision is made by the caller from the schedule.
        """
        n = end - start
        if n <= 0:
            return
        device = new_positions.device
        dtype = new_positions.dtype
        section = ids_list[start:end] if ids_list is not None else None
        pos = 0

        # A chunk that starts mid-block is the continuation of a
        # boundary-spanning block, so the encoder re-encoded it: the FIFO
        # head may be a stale re-encode.  `_resolve_block_geometry` skips it
        # once, only when this flag is set.
        layout["_stale_possible"] = layout.get("phase") in ("vae", "vit")

        def _block_anchor(layout: dict[str, object]) -> int:
            return int(layout.get("next_logical", 0))

        while pos < n:
            phase = layout.get("phase")
            if phase in ("vae", "vit"):
                remaining = int(layout.get("_remaining", 0))
                anchor = int(layout.get("_vae_anchor" if phase == "vae" else "_vit_anchor", 0))
                take = min(remaining, n - pos)
                if take > 0:
                    new_positions[start + pos : start + pos + take] = anchor
                    if phase == "vae":
                        # Mark only the fim-patch interior (markers excluded).
                        for i in range(take):
                            tok = section[pos + i] if section is not None else self._img2img_token_id
                            if tok == self._img2img_token_id:
                                vae_mask[start + pos + i] = True
                pos += take
                remaining -= take
                layout["_remaining"] = remaining
                if phase == "vae":
                    layout["_has_vae"] = True
                if remaining <= 0:
                    if phase == "vae":
                        # VAE section complete -> enter ViT section at M+1.
                        layout["phase"] = "vit"
                        layout["_vit_anchor"] = anchor + 1
                        layout["_remaining"] = int(layout.get("_num_vit", 0))
                    else:
                        # ViT section complete -> block done, text resumes at
                        # M+2 with next_text synced to next_logical.
                        layout["_last_completed_geometry"] = (
                            int(layout.get("_num_vae", 0)),
                            int(layout.get("_num_vit", 0)),
                        )
                        layout["next_logical"] = int(layout.get("next_logical", 0)) + 2
                        layout["next_text"] = layout["next_logical"]
                        layout["phase"] = "text"
                        layout["_vae_anchor"] = None
                        layout["_vit_anchor"] = None
                        # The split-block bindings were consumed; any leftover
                        # geometry is stale and must not satisfy a later
                        # resolve (it would shadow a fresh FIFO entry for a
                        # different-size next image).
                        layout["_block_info"] = None
                continue

            # phase == "text"
            if section is None:
                # No ids (warmup): consume everything as text.
                t0 = int(layout.get("next_text", 0))
                new_positions[start + pos : end] = torch.arange(t0, t0 + (n - pos), device=device, dtype=dtype)
                layout["next_text"] = t0 + (n - pos)
                layout["next_logical"] = t0 + (n - pos)
                pos = n
                break

            # Consume text until the next SOI (block start) or end.
            nxt = pos
            while nxt < n and section[nxt] != self._start_of_image_id:
                nxt += 1
            span_len = nxt - pos
            if span_len > 0:
                t0 = int(layout.get("next_text", 0))
                new_positions[start + pos : start + nxt] = torch.arange(t0, t0 + span_len, device=device, dtype=dtype)
                layout["next_text"] = t0 + span_len
                layout["next_logical"] = t0 + span_len
            pos = nxt
            if pos >= n:
                break
            # section[pos] == SOI -> a block starts here.
            num_vae, num_vit = self._resolve_block_geometry(layout, info_list, info_holder, start + pos)
            if num_vae is None:
                # No geometry for this block: leave as sequential (safety).
                layout["next_text"] = int(layout.get("next_text", 0)) + 1
                layout["next_logical"] = int(layout.get("next_logical", 0)) + 1
                pos += 1
                continue
            layout["phase"] = "vae"
            layout["_vae_anchor"] = _block_anchor(layout)
            layout["_vit_anchor"] = _block_anchor(layout) + 1
            layout["_num_vae"] = num_vae
            layout["_num_vit"] = num_vit
            layout["_remaining"] = num_vae
            # (loop continues; next iteration consumes the VAE section)
            continue

    def _emit_layout_rope(
        self,
        layout: dict[str, object],
        n_computed: int,
        n_scheduled: int,
        done: bool,
    ) -> None:
        """Append the ropes-pending entry for one segment in batch order
        (1:1 with req order -- flush_pending_metadata maps index -> req_id).

        Prefill segments emit the plain rope only; the FIRST DECODE step
        (``done``) adds ``image_shape`` + ``prefill_position_count`` -
        ``num_computed`` equals the true prompt length there.
        ``flush_pending_metadata`` last-wins and its image_shape guard keep
        that entry authoritative across the later plain decode ropes.
        """
        rope = int(layout.get("next_logical", 0))
        if done:
            self._ropes_pending.append(
                {
                    "ropes": [rope],
                    "image_shape": list(layout["_last_img_shape"]),
                    "prefill_position_count": int(n_computed),
                }
            )
        else:
            self._ropes_pending.append({"ropes": [rope]})
