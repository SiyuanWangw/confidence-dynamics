"""Model-family detection, chat templates, and think-tag stripping."""

from __future__ import annotations

import os
from typing import Optional, Tuple

FAMILY_ENGINE = {
    "qwen3.8": {"reasoning_parser": "qwen3"},
    "qwen3.5": {"reasoning_parser": "qwen3"},
    "qwen3": {"reasoning_parser": "qwen3"},
    "gemma4": {"reasoning_parser": "gemma4"},
    # Model card serves this checkpoint with an fp32 Mamba SSM cache.
    "nemotron3": {"reasoning_parser": "nemotron_3", "mamba_ssm_dtype": "float32"},
}
# Gemma 4 docs: T=1.0, top_p=0.95, top_k=64. Qwen thinking runs keep presence_penalty=1.5.
# Nemotron 3 Nano reasoning: T=1.0, top_p=0.95, no top_k, no presence penalty.
FAMILY_SAMPLING = {
    "gemma4": {"top_k": 64, "presence_penalty": 0.0, "skip_special_tokens": False},
    "nemotron3": {"top_k": -1, "presence_penalty": 0.0},
}
GEMMA4_THINK_END = "<channel|>"
QWEN_THINK_END = "</think>"


def _think_end_token(family: str) -> str:
    if family == "gemma4":
        return GEMMA4_THINK_END
    return QWEN_THINK_END


def _strip_think(text: str, family: str = "") -> str:
    token = _think_end_token(family)
    if token in text:
        return text.rpartition(token)[2]
    for alt in (QWEN_THINK_END, GEMMA4_THINK_END):
        if alt != token and alt in text:
            return text.rpartition(alt)[2]
    return text


def _think_answer_lens(
    text: str, tokenizer, think_end: str = QWEN_THINK_END
) -> Tuple[int, int]:
    if think_end in text:
        think, _, ans = text.partition(think_end)
        return len(tokenizer.encode(think)), len(tokenizer.encode(ans))
    return len(tokenizer.encode(text)), 0


def detect_family(model_path: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    blob = os.path.basename(os.path.abspath(model_path)).lower().replace("_", ".")
    if "gemma-4" in blob or "gemma4" in blob:
        return "gemma4"
    if "nemotron" in blob:
        return "nemotron3"
    if "qwen3.8" in blob or "qwen3-8" in blob:
        return "qwen3.8"
    if "qwen3.5" in blob:
        return "qwen3.5"
    if "qwen3" in blob:
        return "qwen3"
    return "generic"


def _apply_family_sampling_defaults(args, family: str) -> None:
    spec = FAMILY_SAMPLING.get(family, {})
    if getattr(args, "top_k", None) is None:
        args.top_k = int(spec.get("top_k", 20))
    if getattr(args, "presence_penalty", None) is None:
        args.presence_penalty = float(spec.get("presence_penalty", 1.5))


def _apply_chat(tokenizer, user_content, family: str) -> str:
    """Apply chat template. ``user_content`` is a string or multimodal part list.

    Base checkpoints such as NVIDIA-Nemotron-Nano-12B-v2-Base ship no
    ``chat_template``. Those prompts stay as the raw user string.
    """
    user = [{"role": "user", "content": user_content}]
    kw = dict(add_generation_prompt=True, tokenize=False)
    try:
        if family.startswith("qwen") or family in ("gemma4", "nemotron3"):
            try:
                return tokenizer.apply_chat_template(user, enable_thinking=True, **kw)
            except TypeError:
                pass
        if family == "gemma4":
            # Docs: thinking is on when <|think|> starts the system prompt.
            chat = [
                {"role": "system", "content": "<|think|>"},
                {"role": "user", "content": user_content},
            ]
            return tokenizer.apply_chat_template(chat, **kw)
        return tokenizer.apply_chat_template(user, **kw)
    except ValueError as exc:
        if "chat_template" not in str(exc) or not isinstance(user_content, str):
            raise
        return user_content
