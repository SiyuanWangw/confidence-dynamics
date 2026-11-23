"""Gemma 4 chat / sampling defaults for over-generate inference."""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from confdyn.generate import (  # noqa: E402
    COUNTDOWN_TEMPLATE_GEMMA4,
    FAMILY_SAMPLING,
    GEMMA4_THINK_END,
    _apply_chat,
    _apply_family_sampling_defaults,
    _sampling_params,
    _strip_think,
    _think_answer_lens,
    _think_end_token,
    _user_prompt,
    detect_family,
)


class _Tok:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, enable_thinking=False, **_kw):
        parts = []
        if enable_thinking:
            parts.append("<|turn>system\n<|think|>\n<turn|>\n")
        for m in messages:
            parts.append(f"<|turn>{m['role']}\n{m['content']}<turn|>\n")
        parts.append("<|turn>model\n")
        if not enable_thinking:
            parts.append("<|channel>thought\n<channel|>")
        return "".join(parts)


class _TokNoEnableKw:
    def apply_chat_template(self, messages, **kw):
        if "enable_thinking" in kw:
            raise TypeError("unexpected kwargs")
        roles = [m["role"] for m in messages]
        assert roles[0] == "system"
        assert messages[0]["content"].startswith("<|think|>")
        return "fallback-system-think"


def test_detect_family_gemma4_12b():
    assert detect_family("/models/gemma-4-12B") == "gemma4"
    assert detect_family("/models/gemma-4-12B-it") == "gemma4"
    assert detect_family("/models/gemma-4-E4B-it") == "gemma4"
    assert detect_family("/models/Qwen3.5-4B") == "qwen3.5"


def test_family_sampling_gemma4():
    args = argparse.Namespace(top_k=None, presence_penalty=None)
    _apply_family_sampling_defaults(args, "gemma4")
    assert args.top_k == 64
    assert args.presence_penalty == 0.0
    sp = _sampling_params(
        argparse.Namespace(
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            min_p=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            max_new_tokens=256,
        ),
        "gemma4",
    )
    assert sp["top_k"] == FAMILY_SAMPLING["gemma4"]["top_k"]
    assert sp["skip_special_tokens"] is False


def test_qwen_sampling_unchanged():
    args = argparse.Namespace(top_k=None, presence_penalty=None)
    _apply_family_sampling_defaults(args, "qwen3.5")
    assert args.top_k == 20
    assert args.presence_penalty == 1.5
    sp = _sampling_params(
        argparse.Namespace(
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            presence_penalty=1.5,
            repetition_penalty=1.0,
            max_new_tokens=256,
        ),
        "qwen3.5",
    )
    assert "skip_special_tokens" not in sp


def test_apply_chat_gemma4_enable_thinking():
    prompt = _apply_chat(_Tok(), "2+2?", "gemma4")
    assert prompt.startswith("<|turn>system\n<|think|>")
    assert "<|turn>user\n2+2?" in prompt
    assert prompt.endswith("<|turn>model\n")
    assert "<|channel>thought\n<channel|>" not in prompt


def test_apply_chat_gemma4_system_fallback():
    assert _apply_chat(_TokNoEnableKw(), "2+2?", "gemma4") == "fallback-system-think"


def test_strip_think_gemma4():
    text = "<|channel>thought\nscratch boxed 1<channel|>\nfinal \\boxed{2}"
    assert _strip_think(text, "gemma4").strip() == "final \\boxed{2}"
    assert _think_end_token("gemma4") == GEMMA4_THINK_END
    tok = _Tok()
    think_n, ans_n = _think_answer_lens(text, tok, GEMMA4_THINK_END)
    assert think_n > 0 and ans_n > 0


def test_countdown_prompt_drops_qwen_think_tags():
    sample = {"prompt": "Make 10 with 1 2 3 4 5 6"}
    text = _user_prompt("countdown6", sample, "gemma4")
    assert "<think>" not in text
    assert "Make 10" in text
    assert "\\boxed" in COUNTDOWN_TEMPLATE_GEMMA4