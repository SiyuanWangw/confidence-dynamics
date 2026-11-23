"""MMMU-Pro (standard 10-option) prompt / image helpers for Gemma4-style VLMs.

Prompt instruction matches GPQA / EvalScope ``ANSWER: [LETTER]``. Images are
interleaved at ``<image N>`` markers when present; otherwise prepended. Chat
templating inserts Gemma4's ``<|image|>`` placeholders; absolute image paths
are passed to ``Engine.generate(..., image_data=...)``.
"""

from __future__ import annotations

import os
import re
from typing import Any, List, Tuple

from confdyn.evals import gpqa_eval

_IMAGE_TAG = re.compile(r"<image\s+(\d+)\s*>", re.IGNORECASE)
IMAGE_PLACEHOLDER = "<|image|>"


def _repo_datasets_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))


def resolve_image_paths(sample: dict, data_root: str | None = None) -> List[str]:
    """Resolve ``images`` entries; paths in the JSON are relative to ``datasets/``."""
    root = data_root or _repo_datasets_root()
    paths: List[str] = []
    for rel in sample.get("images") or []:
        if os.path.isabs(rel):
            p = rel
        else:
            p = os.path.join(root, rel)
            if not os.path.isfile(p):
                p = os.path.join(root, "mmmu_pro150", os.path.basename(rel))
        if not os.path.isfile(p):
            raise FileNotFoundError(f"MMMU image missing: {rel} (resolved {p})")
        paths.append(os.path.abspath(p))
    if not paths:
        raise ValueError(f"sample {sample.get('id')} has no images")
    return paths


def format_question(sample: dict) -> str:
    """Stem (+ optional inline ``<image N>`` tags) with A) … J) options."""
    return gpqa_eval.format_gpqa_question(sample)


def query_text(sample: dict) -> str:
    letters = gpqa_eval.choice_letters(sample)
    return gpqa_eval.query_template(letters).format(Question=format_question(sample))


def _split_by_image_tags(text: str) -> List[Tuple[str, int | None]]:
    """Return [(text_chunk, image_index_or_None), ...] with 1-based indices."""
    parts: List[Tuple[str, int | None]] = []
    last = 0
    for m in _IMAGE_TAG.finditer(text):
        if m.start() > last:
            parts.append((text[last : m.start()], None))
        parts.append(("", int(m.group(1))))
        last = m.end()
    if last < len(text):
        parts.append((text[last:], None))
    if not parts:
        parts.append((text, None))
    return parts


def build_user_content(
    sample: dict, data_root: str | None = None
) -> Tuple[List[dict[str, Any]], List[str]]:
    """OpenAI-style multimodal user content + absolute image paths (in order)."""
    paths = resolve_image_paths(sample, data_root=data_root)
    by_idx = {i + 1: p for i, p in enumerate(paths)}
    text = query_text(sample)
    chunks = _split_by_image_tags(text)
    used_tags = any(idx is not None for _, idx in chunks)

    content: List[dict[str, Any]] = []
    ordered_paths: List[str] = []

    if not used_tags:
        for p in paths:
            content.append({"type": "image"})
            ordered_paths.append(p)
        content.append({"type": "text", "text": text})
        return content, ordered_paths

    for chunk, idx in chunks:
        if chunk:
            content.append({"type": "text", "text": chunk})
        if idx is not None:
            if idx not in by_idx:
                raise KeyError(f"sample {sample.get('id')}: missing image {idx}")
            content.append({"type": "image"})
            ordered_paths.append(by_idx[idx])

    # Append any unused images at the end (rare).
    used = set(ordered_paths)
    for p in paths:
        if p not in used:
            content.append({"type": "image"})
            ordered_paths.append(p)

    if not any(p.get("type") == "text" for p in content):
        content.append({"type": "text", "text": text})
    return content, ordered_paths


def rule_judge(
    solution_str: str, sample: dict, finish_generation: bool = True
) -> tuple[bool, str]:
    letters = gpqa_eval.choice_letters(sample)
    return gpqa_eval.rule_judge(
        solution_str,
        sample.get("final_answer") or sample.get("answer"),
        finish_generation,
        letters=letters,
    )
