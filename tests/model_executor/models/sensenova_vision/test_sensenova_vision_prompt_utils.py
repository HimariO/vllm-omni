# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for :mod:`vllm_omni.model_executor.models.sensenova_vision.prompt_utils`.

The think topology wraps raw user content with the BAGEL think system prompt so
the AR (Thinker) stage decodes ``<thinking>`` tokens before KV transfer.
"""

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.sensenova_vision.prompt_utils import build_think_prompt
from vllm_omni.model_executor.stage_input_processors.bagel import (
    GEN_THINK_SYSTEM_PROMPT,
    VLM_THINK_SYSTEM_PROMPT,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


def test_build_think_prompt_default_image_mode() -> None:
    """Image-output modes use the GEN think system prompt with system+user turns."""
    content = "a cute corgi astronaut"
    out = build_think_prompt(content)

    assert out.startswith(f"{_IM_START}system\n")
    assert GEN_THINK_SYSTEM_PROMPT in out
    assert f"{_IM_END}\n" in out
    assert out.endswith(f"{_IM_START}user\n{content}")
    # The system prompt must be the GEN (image-planning) variant, not the VLM one.
    assert VLM_THINK_SYSTEM_PROMPT not in out


def test_build_think_prompt_think_understanding_mode() -> None:
    """``think_understanding`` uses the VLM think system prompt."""
    content = "what is in this image?"
    out = build_think_prompt(content, mode="think_understanding")

    assert VLM_THINK_SYSTEM_PROMPT in out
    assert GEN_THINK_SYSTEM_PROMPT not in out
    assert out.endswith(f"{_IM_START}user\n{content}")


def test_build_think_prompt_has_expected_markers() -> None:
    """The helper emits exactly the chat markers the tokenizer maps to control ids."""
    out = build_think_prompt("hello")
    assert out.count(_IM_START) == 2  # system + user
    assert out.count(_IM_END) == 1  # closes the system turn
    assert "\n<|im_start|>user\n" in out
