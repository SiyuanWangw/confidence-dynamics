"""Nemotron 3 Nano chat template and reasoning-card sampling defaults."""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from confdyn.generate import (  # noqa: E402
    FAMILY_ENGINE,
    FAMILY_SAMPLING,
    _apply_chat,
    _apply_family_sampling_defaults,
    _sampling_params,
    _strip_think,
    detect_family,
)


class _Tok:
    def apply_chat_template(self, messages, enable_thinking=False, **_kw):
        assert enable_thinking is True
        return "think:" + messages[0]["content"]


def test_detect_family_nemotron3_nano():
    path = "/models/NVIDIA-Nemotron-3-Nano-4B-BF16"
    assert detect_family(path) == "nemotron3"
    assert detect_family("/models/Qwen3-4B") == "qwen3"


def test_family_sampling_nemotron3():
    args = argparse.Namespace(
        temperature=1.0,
        top_p=0.95,
        top_k=None,
        min_p=0.0,
        presence_penalty=None,
        repetition_penalty=1.0,
        max_new_tokens=32,
    )
    _apply_family_sampling_defaults(args, "nemotron3")
    assert args.top_k == -1
    assert args.presence_penalty == 0.0
    sp = _sampling_params(args, "nemotron3")
    assert sp["top_k"] == FAMILY_SAMPLING["nemotron3"]["top_k"]
    assert sp["presence_penalty"] == 0.0
    assert sp["temperature"] == 1.0
    assert sp["top_p"] == 0.95
    assert FAMILY_ENGINE["nemotron3"]["reasoning_parser"] == "nemotron_3"
    assert FAMILY_ENGINE["nemotron3"]["mamba_ssm_dtype"] == "float32"


def test_apply_chat_nemotron3_enable_thinking():
    assert _apply_chat(_Tok(), "2+2?", "nemotron3") == "think:2+2?"


class _BaseTok:
    chat_template = None

    def apply_chat_template(self, messages, **_kw):
        raise ValueError(
            "Cannot use chat template functions because tokenizer.chat_template is not set "
            "and no template argument was passed!"
        )


def test_apply_chat_nemotron_base_without_template():
    assert _apply_chat(_BaseTok(), "2+2?", "nemotron3") == "2+2?"


def test_strip_think_nemotron3():
    text = "<think>\nsteps\n</think>\n\\boxed{4}"
    assert _strip_think(text, "nemotron3").strip() == "\\boxed{4}"
