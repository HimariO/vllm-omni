# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for SenseNova-Vision multi-image support in the AR stage.

Covers four things:

1. ``OmniSenseNovaVisionProcessingInfo`` raises the supported mm limits to
   ``{"image": 10, "img2img": 10}`` (explicit cap matching the upstream
   recon3d ``max_images=10``, kept bounded for mm memory profiling), while
   the shared BAGEL base keeps its conservative ``{...: 1}`` limits.
2. N ``<|image_pad|>`` / ``<|fim_middle|>`` placeholders bind N mm items via
   ``_get_prompt_updates`` expansion + placeholder-range extraction.
3. ``_adjust_positions_for_img2img`` + MoT mask routing is correct for a
   single request containing **two** img2img blocks (this path previously
   consumed only the first block per request).
4. ``embed_multimodal`` returns N embeddings for batched N-item inputs on
   both the ``image`` and ``img2img`` keys.

All tests are CPU-only.  The tokenizer comes from the locally cached
SenseNova-Vision-7B-MoT checkpoint with ``local_files_only=True``; no model
weights are loaded.

Worst-case token budget (stage 0, ``deploy/sensenova_vision.yaml`` has
``max_num_batched_tokens: 32768``)::

    per img2img block (recon3d-size 512x512 input, SenseNova VAE->ViT):
        VAE section  = (512/16)^2 + 2            =   1026 tokens
        separator    =                              1 token
        ViT section  = aspect grid + 2             ~=  1371 tokens
        block total  =                             2398 tokens
    10-image request (limit cap): 10 x 2398       =  23980 prompt tokens

(BAGEL separator layout is kept so extract_embeds_range() yields two mm
ranges for M-RoPE; sizes come from ``_sensenova_*_resize_dims``.)

A single block and a full 10-image request both fit inside one 32768-token
prefill step.  The limit stays a finite 10 (never ``None``) so mm memory
profiling remains bounded.
"""

from __future__ import annotations

import glob
import os
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems

from vllm_omni.diffusion.models.sensenova_vision.tokenization_sensenova_vision import (
    VLLMSenseNovaVisionTokenizer,
)
from vllm_omni.model_executor.models.bagel import bagel as bagel_module
from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
    OmniSenseNovaVisionForConditionalGeneration,
    OmniSenseNovaVisionMultiModalProcessor,
    OmniSenseNovaVisionProcessingInfo,
    _sensenova_img2img_token_counts,
    _sensenova_vae_resize_dims,
    _sensenova_vit_resize_dims,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]

# --- Checkpoint-derived constants (SenseNova-Vision-7B-MoT) -----------------
VIT_MAX_NUM_PATCH_PER_SIDE = 70  # -> 70^2 = 4900 image_pad placeholders/item
VIT_PATCH_TOTAL = VIT_MAX_NUM_PATCH_PER_SIDE**2 + 2  # + start/end markers
LATENT_DOWNSAMPLE = 16  # vae downsample 8 * latent_patch_size 2
MAX_LATENT_SIZE = 64


def _cached_checkpoint() -> str | None:
    """The checkpoint root if cached locally (same lookup as the tokenizer tests)."""
    env_path = os.environ.get("SENSENOVA_VISION_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    snapshot = os.path.expanduser("~/.cache/huggingface/hub/models--sensenova--SenseNova-Vision-7B-MoT/snapshots/*")
    matches = sorted(glob.glob(snapshot))
    return matches[-1] if matches else None


@pytest.fixture(scope="module")
def checkpoint() -> str:
    snap = _cached_checkpoint()
    if snap is None:
        pytest.skip("SenseNova-Vision-7B-MoT not cached and SENSENOVA_VISION_MODEL_PATH is unset")
    assert snap is not None
    return snap


@pytest.fixture(scope="module")
def tokenizer(checkpoint: str) -> VLLMSenseNovaVisionTokenizer:
    """The tokenizer exactly as ``SenseNovaVisionPipeline.__init__`` builds it."""
    return VLLMSenseNovaVisionTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=True,
    )


@pytest.fixture(scope="module")
def hf_config():
    """Faithful stand-in for the merged SenseNovaVision BagelConfig."""
    return SimpleNamespace(
        vit_max_num_patch_per_side=VIT_MAX_NUM_PATCH_PER_SIDE,
        latent_patch_size=2,
        max_latent_size=MAX_LATENT_SIZE,
        vae_config={"downsample": 8, "z_channels": 16},
        vit_config=SimpleNamespace(image_size=980, patch_size=14),
    )


class _StubCtx(SimpleNamespace):
    """Duck-typed ``InputProcessingContext``: only what the processing-info /
    processor paths under test actually touch."""

    def get_tokenizer(self):
        return self.tokenizer

    def get_hf_config(self):
        return self.hf_config


@pytest.fixture()
def info_ctx(tokenizer, hf_config, checkpoint):
    return _StubCtx(
        tokenizer=tokenizer,
        hf_config=hf_config,
        model_config=SimpleNamespace(
            model=checkpoint,
            get_multimodal_config=lambda: SimpleNamespace(enable_mm_embeds=False),
        ),
    )


def _make_processor(info):
    """A real OmniBagelMultiModalProcessor whose info is injected."""
    proc = object.__new__(bagel_module.OmniBagelMultiModalProcessor)
    proc.info = info
    proc.dummy_inputs = None
    proc.cache = None
    proc.data_parser = info.get_data_parser()
    return proc


def _expected_img2img_block_len(h: int, w: int) -> tuple[int, int, int]:
    """(vae_total, vit_total, block_total) for an HxW img2img item.

    Mirrors the BAGEL-BASE resize arithmetic; only used by the 2b test,
    which exercises the base ``OmniBagelMultiModalProcessor`` expansion.
    """
    from vllm_omni.diffusion.models.bagel.pipeline_bagel import bagel_image_size

    stride = LATENT_DOWNSAMPLE
    max_img_size = MAX_LATENT_SIZE * stride
    scale = min(max_img_size / max(h, w), 1.0)
    min_img_size = min(256, max_img_size)
    scale = max(scale, min_img_size / min(h, w))
    new_h = min(max(stride, int(round(h * scale / stride)) * stride), max_img_size)
    new_w = min(max(stride, int(round(w * scale / stride)) * stride), max_img_size)
    num_vae_patches = (new_h // stride) * (new_w // stride)
    num_vae_total = num_vae_patches + 2
    vit_w, vit_h = bagel_image_size(w, h, 980, 224, 14)
    num_vit_total = (vit_h // 14) * (vit_w // 14) + 2
    return num_vae_total, num_vit_total, num_vae_total + 1 + num_vit_total


# ---------------------------------------------------------------------------
# 1. mm limits override
# ---------------------------------------------------------------------------


def test_sensenova_mm_limits_raise_to_ten(info_ctx):
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionProcessingInfo,
    )

    info = OmniSenseNovaVisionProcessingInfo(info_ctx)
    assert info.get_supported_mm_limits() == {"image": 10, "img2img": 10}


def test_model_class_registered_with_sensenova_info():
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionForConditionalGeneration,
        OmniSenseNovaVisionProcessingInfo,
    )

    factories = OmniSenseNovaVisionForConditionalGeneration._processor_factory
    assert factories.info is OmniSenseNovaVisionProcessingInfo


# ---------------------------------------------------------------------------
# 2a. N <|image_pad|> placeholders bind N image items
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2b. N <|fim_middle|> placeholders bind N img2img items -> N blocks
# ---------------------------------------------------------------------------


def test_sensenova_img2img_expansion_is_upstream_exact(tokenizer, hf_config, info_ctx):
    """SenseNova placeholder counts must lockstep with ``_sensenova_*_resize_dims``.

    Layout keeps BAGEL's separator so extract_embeds_range() yields two mm
    ranges for M-RoPE; sizes use the official VAE then ViT chain.
    """
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionMultiModalProcessor,
        OmniSenseNovaVisionProcessingInfo,
    )

    h, w = 375, 500  # non-square -> resize arithmetic actually exercised
    image = Image.new("RGB", (w, h))
    mm_items = MultiModalDataItems({"img2img": bagel_module.Img2ImgProcessorItems([image])})

    info = OmniSenseNovaVisionProcessingInfo(info_ctx)
    proc = object.__new__(OmniSenseNovaVisionMultiModalProcessor)
    proc.info = info
    proc.dummy_inputs = None
    proc.cache = None
    proc.data_parser = info.get_data_parser()

    updates = proc._get_prompt_updates(mm_items, {}, MultiModalKwargsItems())
    mm_prompt_updates = proc._bind_and_group_updates(updates, mm_items.get_all_counts())
    prompt_ids = [tokenizer.convert_tokens_to_ids("<|fim_middle|>")]
    _new_ids, placeholders = proc._apply_prompt_updates(prompt_ids, mm_prompt_updates)

    (ph,) = placeholders["img2img"]
    fim_id = tokenizer.get_vocab()["<|fim_middle|>"]

    # Official two-stage transform: VAE resize, then ViT resize OF THE VAE-
    # RESIZED image.  The ViT count follows aspect ratio (upstream
    # ImageTransform(980, 224, 14)), NOT the fixed 70x70 square.
    new_h, new_w = _sensenova_vae_resize_dims(h, w)
    vit_h, vit_w = _sensenova_vit_resize_dims(new_h, new_w)
    num_vae_total, num_vit_total, vae_h, vae_w = _sensenova_img2img_token_counts(h, w)
    assert (vae_h, vae_w) == (new_h, new_w)
    num_vit_patches = (vit_h // 14) * (vit_w // 14)
    assert num_vit_patches <= VIT_MAX_NUM_PATCH_PER_SIDE**2, "aspect grid must stay within the 70x70 cap"
    assert num_vit_total == num_vit_patches + 2

    # BAGEL separator between VAE and ViT sections for M-RoPE ranges.
    total = num_vae_total + 1 + num_vit_total
    assert ph.length == total
    assert all(t == fim_id for t in ph.tokens)

    mask = ph.is_embed
    assert mask is not None and mask.shape[0] == total
    assert mask[:num_vae_total].all(), "VAE section must be embedded"
    assert not mask[num_vae_total], "separator must not be embedded"
    assert mask[num_vae_total + 1 :].all(), "ViT section must be embedded"


# ---------------------------------------------------------------------------
# 2d. embed_multimodal returns N embeddings for batched N-item inputs
# ---------------------------------------------------------------------------


def test_parse_and_validate_splits_image_and_img2img():
    inst = object.__new__(bagel_module.OmniBagelForConditionalGeneration)
    pv_image = torch.zeros(2, 3, 8, 8)
    pv_img2img = torch.zeros(3, 3, 8, 8)

    mm = inst._parse_and_validate_multimodal_inputs(pixel_values=pv_image, pixel_values_img2img=pv_img2img)

    assert set(mm) == {"img2text", "img2img"}
    assert mm["img2text"]["pixel_values"] is pv_image
    assert mm["img2img"]["pixel_values"] is pv_img2img


def test_embed_multimodal_returns_n_embeddings_per_modality():
    n_images, n_img2img = 3, 2
    inst = object.__new__(bagel_module.OmniBagelForConditionalGeneration)
    calls = []

    def fake_img2text(mm_input):
        calls.append(("img2text", mm_input["pixel_values"].shape[0]))
        return tuple(torch.full((1, 4), float(i)) for i in range(n_images))

    def fake_img2img(mm_input):
        calls.append(("img2img", mm_input["pixel_values"].shape[0]))
        return tuple(torch.full((1, 4), 100.0 + i) for i in range(n_img2img))

    inst._parse_and_validate_multimodal_inputs = lambda **kw: {
        "img2text": {"pixel_values": torch.zeros(n_images, 3, 8, 8)},
        "img2img": {"pixel_values": torch.zeros(n_img2img, 3, 8, 8)},
    }
    inst._process_img2text_input = fake_img2text
    inst._process_img2img_input = fake_img2img

    out = inst.embed_multimodal()

    assert ("img2text", n_images) in calls and ("img2img", n_img2img) in calls
    assert len(out) == n_images + n_img2img, "one embedding per mm item"
    assert [t[0, 0].item() for t in out] == [0, 1, 2, 100, 101]


def test_img2img_batch_flattens_leading_batch_dim():
    """A (B, N, C, H, W) img2img tensor must yield one info tuple per image."""
    inst = object.__new__(bagel_module.OmniBagelForConditionalGeneration)
    infos: list[tuple[int, int, int, int]] = []
    inst.latent_downsample = LATENT_DOWNSAMPLE
    inst.max_latent_size = MAX_LATENT_SIZE
    inst.latent_channel = 16
    inst.latent_patch_size = 2
    inst.config = SimpleNamespace(vit_config=SimpleNamespace(image_size=64, patch_size=14))
    inst.device = torch.device("cpu")

    captured = {}

    def fake_vit_embeddings(images):
        captured["n"] = len(images)
        return [torch.zeros(1, 4) for _ in images]

    class _FakeVAE:
        def encode(self, x):
            # Bare latent tensor, 16 channels, /8 spatial (DiagonalGaussian output).
            return torch.zeros(x.shape[0], 16, x.shape[2] // 8, x.shape[3] // 8)

    inst._vit_embeddings = fake_vit_embeddings
    inst.vae = _FakeVAE()
    inst._resize_to_stride = lambda pv: pv
    inst.get_flattened_position_ids = lambda *a, **k: torch.zeros(1, dtype=torch.long)
    inst.language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids: torch.zeros(len(ids), 4)))
    inst.vae2llm = lambda z: torch.zeros(z.shape[0], 4)
    inst.latent_pos_embed = lambda pos: torch.zeros(1, 4)
    inst.time_embedder = lambda t: torch.zeros(1, 4)
    inst._start_of_image_id = 151652
    inst._end_of_image_id = 151653
    inst._ropes_pending = []
    inst._pending_img2img_info = infos
    inst._img2img_info_by_size = {}
    inst._img2img_by_req = {}
    inst._last_img2img_info = None

    batched = torch.zeros(1, 2, 3, 32, 32)  # (batch=1, num_images=2, ...)
    inst._process_img2img_input({"pixel_values": batched})

    assert captured["n"] == 2, "leading batch dim must be flattened"
    assert len(infos) == 2, "one (num_vae, num_vit, H, W) info tuple per image"


def test_sensenova_img2img_seeds_size_cache_for_cache_served_request():
    """A SenseNova img2img embed must seed the cross-request size cache.

    Regression for the aspect-ratio bug: ``seg`` (2.jpg) then ``normal``
    (2.jpg) in one process lost the second request's ``image_shape``, so the
    DiT fell back to a square 1024x1024 output.  ``_process_img2img_input``
    appends to ``_pending_img2img_info`` but the size lookup for a later
    request whose image the encoder/prefix cache serves (no embed run) is
    ``_img2img_info_by_size`` — that cache is only seeded by
    ``_register_img2img_info``.
    """
    from vllm_omni.model_executor.models.sensenova_vision.sensenova_vision import (
        OmniSenseNovaVisionForConditionalGeneration,
    )

    inst = object.__new__(OmniSenseNovaVisionForConditionalGeneration)
    inst.latent_downsample = LATENT_DOWNSAMPLE
    inst.max_latent_size = MAX_LATENT_SIZE
    inst.latent_channel = 16
    inst.latent_patch_size = 2
    inst.config = SimpleNamespace(vit_config=SimpleNamespace(image_size=64, patch_size=14))
    inst.device = torch.device("cpu")

    captured = {}

    def fake_vit_embeddings(images):
        captured["n"] = len(images)
        return [torch.zeros(1, 4) for _ in images]

    class _FakeVAE:
        def encode(self, x):
            return torch.zeros(x.shape[0], 16, x.shape[2] // 8, x.shape[3] // 8)

    inst._vit_embeddings = fake_vit_embeddings
    inst._resize_to_stride = lambda pv: pv
    inst._resize_for_vit = lambda pv: pv
    inst.vae = _FakeVAE()
    inst.get_flattened_position_ids = lambda *a, **k: torch.zeros(1, dtype=torch.long)
    inst.language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids: torch.zeros(len(ids), 4)))
    inst.vae2llm = lambda z: torch.zeros(z.shape[0], 4)
    inst.latent_pos_embed = lambda pos: torch.zeros(1, 4)
    inst.time_embedder = lambda t: torch.zeros(1, 4)
    inst._start_of_image_id = 151652
    inst._end_of_image_id = 151653
    inst._ropes_pending = []
    inst._pending_img2img_info = []
    inst._img2img_info_by_size = {}
    inst._img2img_by_req = {}
    inst._last_img2img_info = None

    img = torch.zeros(1, 1, 3, 32, 32)  # (batch, num_images, C, H, W)
    inst._process_img2img_input({"pixel_values": img})

    # Pending metadata consumed by this step's routing...
    assert len(inst._pending_img2img_info) == 1
    # ...and the size cache must ALSO be seeded so a cache-served follow-up
    # request (no embed run) can resolve its (H, W).
    key = tuple(inst._pending_img2img_info[0][:2])
    assert key in inst._img2img_info_by_size, "size cache must be seeded by the embed run"
    assert inst._img2img_info_by_size[key][2:] == (32, 32), inst._img2img_info_by_size[key]


# ---------------------------------------------------------------------------
# 3. Worst-case token budget arithmetic
# ---------------------------------------------------------------------------


def test_worst_case_token_budget_arithmetic():
    """Budget for the limit=10 cap with SenseNova VAE/ViT lockstep sizing.

    A 512x512 recon3d-size block uses VAE+sep+ViT placeholders and must fit
    inside one stage-0 prefill step (``max_num_batched_tokens: 32768``).
    With aspect-aware ViT the 10-image budget also fits in one step.
    """

    num_vae, num_vit, _, _ = _sensenova_img2img_token_counts(512, 512)
    per_block = num_vae + 1 + num_vit
    assert per_block == 2398
    assert per_block <= 32768
    assert 10 * per_block <= 32768


def _model_stub() -> OmniSenseNovaVisionForConditionalGeneration:
    inst = object.__new__(OmniSenseNovaVisionForConditionalGeneration)
    inst.latent_downsample = 16
    inst.max_latent_size = 64
    inst.latent_channel = 16
    inst.latent_patch_size = 2
    inst.config = SimpleNamespace(vit_config=SimpleNamespace(image_size=64, patch_size=14))
    inst.device = torch.device("cpu")
    return inst


def test_resize_methods_select_the_right_grid() -> None:
    inst = _model_stub()
    pv = torch.zeros(1, 3, 200, 300)
    assert inst._resize_to_stride(pv).shape[2:] == (512, 768)
    assert inst._resize_to_recon3d_vae(pv).shape[2:] == (256, 384)
    # ViT transforms operate on the already-VAE-resized image.
    assert inst._resize_for_vit(torch.zeros(1, 3, 384, 512)).shape[2:] == (378, 518)
    assert inst._resize_to_recon3d_vit(torch.zeros(1, 3, 384, 512)).shape[2:] == (336, 448)


# ---------------------------------------------------------------------------
# Embed gate: 1 image -> default VAE grid, >1 images -> recon3d VAE grid
# ---------------------------------------------------------------------------


def _wire_embed_fakes(inst, calls: dict) -> None:
    """Attach the fakes ``_process_img2img_input`` needs; record resize calls."""

    def fake_vit_embeddings(images):
        calls["vit_sizes"] = [tuple(img.shape[-2:]) for img in images]
        return [torch.zeros((img.shape[-2] // 14) * (img.shape[-1] // 14), 4) for img in images]

    class _FakeVAE:
        def encode(self, x):
            return torch.zeros(x.shape[0], 16, x.shape[2] // 8, x.shape[3] // 8)

    orig_stride = OmniSenseNovaVisionForConditionalGeneration._resize_to_stride
    orig_recon3d = OmniSenseNovaVisionForConditionalGeneration._resize_to_recon3d_vae
    orig_vit_default = OmniSenseNovaVisionForConditionalGeneration._resize_for_vit
    orig_vit_recon3d = OmniSenseNovaVisionForConditionalGeneration._resize_to_recon3d_vit

    def stride_resize(pv):
        calls.setdefault("stride", []).append(tuple(pv.shape[2:]))
        return orig_stride(inst, pv)

    def recon3d_resize(pv):
        calls.setdefault("recon3d", []).append(tuple(pv.shape[2:]))
        return orig_recon3d(inst, pv)

    def vit_default(pv):
        calls.setdefault("vit_default", []).append(tuple(pv.shape[2:]))
        return orig_vit_default(inst, pv)

    def vit_recon3d(pv):
        calls.setdefault("vit_recon3d", []).append(tuple(pv.shape[2:]))
        return orig_vit_recon3d(inst, pv)

    inst._vit_embeddings = fake_vit_embeddings
    inst._resize_to_stride = stride_resize
    inst._resize_to_recon3d_vae = recon3d_resize
    inst._resize_for_vit = vit_default
    inst._resize_to_recon3d_vit = vit_recon3d
    inst.vae = _FakeVAE()
    inst.get_flattened_position_ids = lambda *a, **k: torch.zeros(1, dtype=torch.long)
    inst.language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=lambda ids: torch.zeros(len(ids), 4)))
    inst.vae2llm = lambda z: torch.zeros(z.shape[0], 4)
    inst.latent_pos_embed = lambda pos: torch.zeros(1, 4)
    inst.time_embedder = lambda t: torch.zeros(1, 4)
    inst._start_of_image_id = 151652
    inst._end_of_image_id = 151653
    inst._ropes_pending = []
    inst._pending_img2img_info = []
    inst._img2img_info_by_size = {}
    inst._img2img_by_req = {}
    inst._last_img2img_info = None


def test_embed_single_image_uses_default_vae_grid() -> None:
    inst = _model_stub()
    calls: dict = {}
    _wire_embed_fakes(inst, calls)

    inst._process_img2img_input({"pixel_values": torch.zeros(1, 1, 3, 200, 300)})

    assert calls.get("recon3d") is None, "single image must not take the recon3d transform"
    assert calls.get("vit_recon3d") is None, "single image must not take the recon3d ViT transform"
    assert calls["stride"] == [(200, 300)]
    # ViT sees the default-chain dims of the VAE-resized image.
    assert calls["vit_default"] == [(512, 768)]
    # info (h_px, w_px) follows the default grid and feeds kv_metadata["image_shape"].
    infos = list(inst._img2img_info_by_size.values())
    assert len(infos) == 1 and infos[0][2:] == (512, 768)
    assert len(inst._pending_img2img_info) == 1


def test_embed_multi_image_uses_recon3d_vae_grid() -> None:
    inst = _model_stub()
    calls: dict = {}
    _wire_embed_fakes(inst, calls)

    inst._process_img2img_input({"pixel_values": torch.zeros(1, 2, 3, 200, 300)})

    assert calls.get("stride") is None, "multi-view must not take the default transform"
    assert calls.get("vit_default") is None, "multi-view must not take the default ViT transform"
    assert calls["recon3d"] == [(200, 300)] * 2
    # ViT sees the recon3d-chain dims of the VAE-resized image.
    assert calls["vit_recon3d"] == [(256, 384)] * 2
    assert calls["vit_sizes"] == [(252, 378)] * 2
    # Every view's info carries the recon3d VAE dims -> DiT image_shape.  The
    # size cache is keyed by (num_vae, num_vit), so two identical views
    # collapse to one entry (base-class dedup); the pending list stays 1/image.
    infos = list(inst._img2img_info_by_size.values())
    assert len(infos) == 1 and infos[0][2:] == (256, 384)
    assert len(inst._pending_img2img_info) == 2
    assert all(info[2:] == (256, 384) for info in inst._pending_img2img_info)


# ---------------------------------------------------------------------------
# Processor parity: placeholder counts must match the embed-side gate
# ---------------------------------------------------------------------------


def _img2img_placeholder_lengths(proc, tokenizer, sizes_w_h: list[tuple[int, int]]) -> list[int]:
    images = [Image.new("RGB", size) for size in sizes_w_h]
    mm_items = MultiModalDataItems({"img2img": bagel_module.Img2ImgProcessorItems(images)})
    updates = proc._get_prompt_updates(mm_items, {}, MultiModalKwargsItems())
    mm_prompt_updates = proc._bind_and_group_updates(updates, mm_items.get_all_counts())
    prompt_ids = [tokenizer.convert_tokens_to_ids("<|fim_middle|>")] * len(sizes_w_h)
    _new_ids, placeholders = proc._apply_prompt_updates(prompt_ids, mm_prompt_updates)
    blocks = placeholders["img2img"]
    assert [ph.item_idx for ph in blocks] == list(range(len(sizes_w_h)))
    return [ph.length for ph in blocks]


@pytest.mark.parametrize("num_images", [1, 2])
def test_placeholder_matches_runtime_embeddings(num_images, tokenizer, info_ctx):
    info = OmniSenseNovaVisionProcessingInfo(info_ctx)
    proc = object.__new__(OmniSenseNovaVisionMultiModalProcessor)
    proc.info = info
    proc.dummy_inputs = None
    proc.cache = None
    proc.data_parser = info.get_data_parser()
    model = _model_stub()
    calls: dict[str, list[tuple[int, int]]] = {}
    _wire_embed_fakes(model, calls)
    embeddings = model._process_img2img_input({"pixel_values": torch.zeros(1, num_images, 3, 200, 300)})
    lengths = _img2img_placeholder_lengths(proc, tokenizer, [(300, 200)] * num_images)
    # The separator token remains text, outside the VAE and ViT embedding ranges.
    assert lengths == [embedding.shape[0] + 1 for embedding in embeddings]
