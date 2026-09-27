# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for SenseNova topology, text routing, and offline caption output."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image
from vllm.outputs import CompletionOutput, RequestOutput

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.diffusion import output_formatter
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.models.sensenova_vision import single_stage as snv_single_stage
from vllm_omni.diffusion.models.sensenova_vision.pipeline_sensenova_vision import (
    SenseNovaVisionPipeline,
    build_sensenova_vision_diffusion_output,
)
from vllm_omni.diffusion.output_formatter import (
    format_diffusion_outputs,
    normalize_diffusion_postprocess_output,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.sensenova_vision.pipeline import (
    SENSENOVA_VISION_PIPELINE,
    SENSENOVA_VISION_SINGLE_STAGE_PIPELINE,
    SENSENOVA_VISION_THINK_PIPELINE,
)
from vllm_omni.model_executor.models.sensenova_vision.prompt_utils import (
    bridge_think_text_to_image,
)

_CAPTION = "a red car parked in front of a brick wall"

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def _pipeline() -> SenseNovaVisionPipeline:
    return object.__new__(SenseNovaVisionPipeline)


def _contexts() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], tuple[int, int]]:
    cache = SimpleNamespace(seq_lens=[8])
    context = {"kv_lens": [8], "ropes": [8], "past_key_values": cache}
    return context, dict(context), dict(context), (16, 16)


def test_single_stage_context_clone_isolates_branch_owned_state() -> None:
    class FakeCache:
        def __init__(self, layers: list[int]) -> None:
            self.layers = layers

        def copy(self) -> FakeCache:
            return FakeCache(list(self.layers))

    context: dict[str, Any] = {"kv_lens": [8], "ropes": [12], "past_key_values": FakeCache([1])}
    clone = snv_single_stage._clone_single_stage_context(context)
    clone["kv_lens"].append(4)
    clone["ropes"][0] = 99
    clone["past_key_values"].layers.append(2)

    assert context["kv_lens"] == [8]
    assert context["ropes"] == [12]
    assert clone["past_key_values"] is not context["past_key_values"]
    assert context["past_key_values"].layers == [1]


def test_single_stage_img2text_decodes_from_sensenova_local_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text output does not fall back to BAGEL's generic image prefill."""
    pipeline = _pipeline()
    contexts = _contexts()
    calls: dict[str, Any] = {}
    monkeypatch.setattr(
        pipeline,
        "_prepare_single_stage_contexts",
        lambda prompt, sampling: calls.setdefault("prepared", contexts),
    )
    monkeypatch.setattr(
        pipeline,
        "_decode_single_stage_text",
        lambda context, sampling: calls.setdefault("decoded", "a giraffe beside a fence"),
    )

    output = pipeline._forward_single(
        {"prompt": "describe", "modalities": ["text"]},
        OmniDiffusionSamplingParams(extra_args={"max_think_tokens": 8192}),
    )

    assert calls["prepared"] is contexts
    assert calls["decoded"] == "a giraffe beside a fence"
    assert output.output["payload"] == {"text": "a giraffe beside a fence"}


def test_single_stage_caption_generate_decodes_before_injected_kv_denoising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thinking image modes retain their locally decoded caption for merging."""
    pipeline = _pipeline()
    contexts = _contexts()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(pipeline, "_prepare_single_stage_contexts", lambda prompt, sampling: contexts)
    monkeypatch.setattr(pipeline, "_decode_single_stage_text", lambda context, sampling: "interleaved caption")
    monkeypatch.setattr(
        pipeline,
        "_update_single_stage_text_context",
        lambda context, text: captured.setdefault("reencoded", (context, text)),
    )

    def fake_base_forward(self, prompt, sampling, *, prepare_only=False):
        captured["sampling"] = sampling
        return DiffusionOutput(output={"payload": {"image": "image"}})

    monkeypatch.setattr(BagelPipeline, "_forward_single", fake_base_forward)
    sampling = OmniDiffusionSamplingParams(extra_args={"think": True, "max_think_tokens": 8192})

    output = pipeline._forward_single(
        {"prompt": "caption", "modalities": ["img2img"]},
        sampling,
    )

    assert output.output["payload"] == {"image": "image"}
    assert captured["sampling"].past_key_values is contexts[0]["past_key_values"]
    assert captured["reencoded"] == (contexts[0], "interleaved caption")
    assert sampling.extra_args["text_output"] == "interleaved caption"


def test_single_stage_prefill_normalizes_transport_markers_and_uses_vit_only_for_understanding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Understanding converts the two-stage prompt into upstream raw terms."""

    class FakeCache:
        def __init__(self, _layers: int) -> None:
            self.seq_lens = [0]

        def copy(self) -> FakeCache:
            return FakeCache(0)

    calls: dict[str, Any] = {"vae": 0, "vit": 0, "texts": []}
    pipeline = _pipeline()
    pipeline.tokenizer = SimpleNamespace()
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace(dtype=torch.bfloat16)
    pipeline.new_token_ids = {}
    pipeline.bagel = SimpleNamespace(
        config=SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=1)),
        max_latent_size=64,
        latent_downsample=8,
        prepare_prompts=lambda curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids: (
            calls["texts"].append(prompts[0]) or {"text": torch.tensor([1])},
            [curr_kvlens[0] + 3],
            [curr_rope[0] + 3],
        ),
        prepare_vit_images=lambda curr_kvlens, curr_rope, images, transforms, new_token_ids: (
            (calls.update({"vit": calls["vit"] + 1}) or {"vit": torch.tensor([1])}),
            [curr_kvlens[0] + 3],
            [curr_rope[0] + 1],
        ),
        forward_cache_update_vit=lambda cache, **kwargs: cache,
        forward_cache_update_text=lambda cache, **kwargs: cache,
        prepare_vae_images=lambda *args, **kwargs: calls.update({"vae": calls["vae"] + 1}),
    )
    pipeline._resize_context_image = lambda image, **kwargs: image
    pipeline._context_vit_transform = lambda image, **kwargs: torch.zeros(3, 14, 14)
    monkeypatch.setattr(snv_single_stage, "NaiveCache", FakeCache)

    prompt = "<|im_start|>user\n<|image_pad|>\nfind birds<|im_end|>\n<|im_start|>assistant\n"
    pipeline._prepare_single_stage_contexts(
        {"prompt": prompt, "multi_modal_data": {"image": Image.new("RGB", (16, 16))}},
        OmniDiffusionSamplingParams(),
    )

    # Positive and image-CFG contexts both receive only the raw user term;
    # Bagel.prepare_prompts owns its BOS/EOS wrapper.
    assert calls["texts"] == ["find birds", "find birds"]
    assert calls["vae"] == 0
    assert calls["vit"] == 1


@pytest.mark.parametrize(
    "name, config, thinking",
    [
        ("sensenova_vision", SENSENOVA_VISION_PIPELINE, False),
        ("sensenova_vision_think", SENSENOVA_VISION_THINK_PIPELINE, True),
        ("sensenova_vision_single_stage", SENSENOVA_VISION_SINGLE_STAGE_PIPELINE, None),
    ],
)
def test_pipeline_topology(name, config, thinking):
    assert OMNI_PIPELINES[name] is config
    assert resolve_pipeline_config(name) is config
    assert config.default_deploy_config_name == name + ".yaml"
    stage0 = config.get_stage(0)
    assert stage0 is not None
    if thinking is None:
        assert stage0.execution_type == StageExecutionType.DIFFUSION
        assert stage0.input_sources == ()
        assert config.get_stage(1) is None
        return
    kv = stage0.omni_kv_config or {}
    assert kv["need_send_cache"] is True
    if thinking:
        assert stage0.prompt_expand_func.endswith(".expand_sensenova_cfg_prompts_think")
        assert "kv_transfer_criteria" not in kv
    else:
        assert kv["kv_transfer_criteria"] == {"type": "prefill_finished"}
    stage1 = config.get_stage(1)
    assert stage1.execution_type == StageExecutionType.DIFFUSION
    assert stage1.input_sources == (0,)
    assert stage1.omni_kv_config["need_recv_cache"] is True
    if thinking:
        assert stage1.custom_process_input_func.endswith(".bridge_think_text_to_image")


def _source_outputs(text: str) -> list[SimpleNamespace]:
    completion = CompletionOutput(
        index=0,
        text=text,
        token_ids=[1, 2, 3],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="length",
        stop_reason=None,
    )
    return [
        RequestOutput(
            request_id="req-ar",
            prompt="prompt",
            prompt_token_ids=[101],
            prompt_logprobs=None,
            outputs=[completion],
            finished=True,
            metrics=None,
            lora_request=None,
        )
    ]


@pytest.mark.parametrize(
    "mode, thinking",
    [
        ("caption_generate", True),
        ("think_generate", True),
        ("dense_perception", False),
        ("recon3d", False),
    ],
)
def test_bridge_and_merge_caption_output(mode, thinking):
    params = OmniDiffusionSamplingParams(extra_args={"think": thinking})
    prompt = {"mode": mode}
    assert bridge_think_text_to_image(_source_outputs("caption"), prompt=prompt, sampling_params=params) is prompt
    assert params.extra_args.get("text_output") == ("caption" if thinking else None)
    # Exercise the DiT gate even when a non-thinking request contains stray AR text.
    params.extra_args["text_output"] = "caption"
    output = DiffusionOutput(output={"payload": {"image": Image.new("RGB", (8, 8))}, "metadata": {}})
    merged = _pipeline()._merge_mixed_task_text(SimpleNamespace(sampling_params=params), output)
    if thinking:
        assert merged.output["payload"]["text"] == "caption"
        assert merged.output["metadata"]["text"]["text_output"] == "caption"
    else:
        assert merged is output
        assert "text" not in output.output["payload"]


@pytest.mark.parametrize(
    "prompt, params, outputs",
    [
        ("bare string", OmniDiffusionSamplingParams(), _source_outputs("text")),
        ({"mode": "caption_generate"}, None, _source_outputs("text")),
        ({"mode": "caption_generate"}, OmniDiffusionSamplingParams(), []),
    ],
)
def test_bridge_without_caption_is_passthrough(prompt, params, outputs):
    assert bridge_think_text_to_image(outputs, prompt=prompt, sampling_params=params) is prompt
    if params is not None:
        assert "text_output" not in params.extra_args


def test_merge_prefers_metadata_caption():
    image = _image()
    output = DiffusionOutput(output={"payload": {"image": image}, "metadata": {"text": {"think_text": "caption"}}})
    params = OmniDiffusionSamplingParams(extra_args={"think": True, "text_output": "fallback"})
    merged = _pipeline()._merge_mixed_task_text(SimpleNamespace(sampling_params=params), output)
    assert merged.output["payload"] == {"image": image, "text": "caption"}
    assert merged.output["metadata"]["text"] == {"think_text": "caption", "text_output": "caption"}


def _image() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(200, 40, 40))


def _request() -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt="generate an image",
        request_id="req-mixed",
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=1,
            num_outputs_per_prompt=1,
            resolution=512,
        ),
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(model_class_name="SenseNovaVisionPipeline")


def test_mixed_payload_formats_to_image_with_caption_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed payload formats as an image with offline caption metadata.

    The shared formatter does not promote secondary text to a top-level
    multimodal output. Offline consumers recover it from ``metadata.text``.
    """
    monkeypatch.setattr(output_formatter, "supports_audio_output", lambda _: False)
    image = _image()
    diffusion_output = build_sensenova_vision_diffusion_output(
        text=_CAPTION,
        image=image,
        think_text="thinking before caption",
    )
    postprocess_output = normalize_diffusion_postprocess_output(diffusion_output.output)

    assert postprocess_output.primary_key == "image"
    assert postprocess_output.outputs == {"text": _CAPTION, "image": image}

    [result] = format_diffusion_outputs(
        request=_request(),
        od_config=_config(),
        diffusion_output=diffusion_output,
        output_data=diffusion_output.output,
        postprocess_output=postprocess_output,
    )

    assert result.images == [image]
    assert result.final_output_type == "image"
    assert "text" not in result.multimodal_output
    assert result.multimodal_output["metadata"]["text"] == {
        "text_output": _CAPTION,
        "think_text": "thinking before caption",
    }
    # The multimodal output must not gain any SenseNovaVision-specific modality keys.
    assert not ({"depth", "normal", "segmentation", "camera_pose", "point_map"} & set(result.multimodal_output))


def test_merge_is_additive_when_text_already_present() -> None:
    """Outputs that already carry a text payload are returned unchanged."""
    pipeline = _pipeline()
    existing = DiffusionOutput(
        output={
            "payload": {"text": "keep me", "image": _image()},
            "metadata": {"text": {"text_output": "keep me"}},
        }
    )
    req = DiffusionRequestBatch(requests=[_request()])

    merged = pipeline._merge_mixed_task_text(req, existing)

    assert merged is existing
    assert merged.output["payload"]["text"] == "keep me"


def _merged_output(think_text: str | None = None) -> DiffusionOutput:
    payload: dict[str, object] = {"image": _image()}
    metadata: dict[str, object] = {}
    if think_text is not None:
        metadata["text"] = {"think_text": think_text}
    return DiffusionOutput(output={"payload": payload, "metadata": metadata}, stage_durations={"execute": 0.5})


def test_merge_leaves_payload_unchanged_without_text() -> None:
    """No caption available anywhere -> the image-only payload is unchanged."""
    output = _merged_output()
    req = DiffusionRequestBatch(requests=[_request()])

    merged = _pipeline()._merge_mixed_task_text(req, output)

    assert merged is output
    assert "text" not in merged.output["payload"]
    assert merged.output["metadata"] == {}
