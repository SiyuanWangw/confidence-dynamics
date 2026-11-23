"""CPU checks for the MMMU-Pro 150-item multimodal subset."""

import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from confdyn.evals import mmmu_eval
from confdyn.generate import (
    DEFAULT_MMMU_PRO150,
    VLM_MCQ_DATASETS,
    _dataset_defaults,
    _judge_sample,
    _load_samples,
    _user_prompt,
)


def test_mmmu_pro150_is_vlm_mcq():
    assert "mmmu_pro150" in VLM_MCQ_DATASETS
    assert _dataset_defaults("mmmu_pro150") == DEFAULT_MMMU_PRO150


def test_mmmu_pro150_has_one_fifty():
    rows = _load_samples(DEFAULT_MMMU_PRO150)
    assert len(rows) == 150
    counts = Counter(r["subject"] for r in rows)
    assert len(counts) == 30
    assert all(v == 5 for v in counts.values())
    assert rows[0]["id"] == "mmmu_pro150_000"
    assert rows[-1]["id"] == "mmmu_pro150_149"
    assert all(r.get("n_options") == 10 for r in rows)
    assert all(len(r.get("images") or []) >= 1 for r in rows)
    for r in rows[:3]:
        paths = mmmu_eval.resolve_image_paths(r)
        assert all(os.path.isfile(p) for p in paths)


def test_prompt_is_ten_option_mcq():
    rows = _load_samples(DEFAULT_MMMU_PRO150)
    prompt = _user_prompt("mmmu_pro150", rows[0])
    assert "ANSWER: [LETTER]" in prompt
    assert "A, B, C, D, E, F, G, H, I, J" in prompt
    for letter in "ABCDEFGHIJ":
        assert f"{letter})" in prompt or f"{letter}." in mmmu_eval.format_question(
            rows[0]
        )


def test_build_user_content_has_images():
    rows = _load_samples(DEFAULT_MMMU_PRO150)
    content, paths = mmmu_eval.build_user_content(rows[0])
    assert paths
    assert any(p.get("type") == "image" for p in content)
    assert any(p.get("type") == "text" for p in content)
    assert content[-1]["type"] == "text" or any(p["type"] == "text" for p in content)


def test_letter_judge():
    sample = {
        "kind": "mmmu_pro",
        "final_answer": "C",
        "n_options": 10,
        "options": [str(i) for i in range(10)],
        "choices": {chr(ord("A") + i): str(i) for i in range(10)},
    }
    ok, extracted = _judge_sample(
        "mmmu_pro150", "Reasoning.\nANSWER: C", sample, True, "gemma4"
    )
    assert ok, extracted
    ok, _ = _judge_sample(
        "mmmu_pro150", "ANSWER: A", sample, True, "gemma4"
    )
    assert not ok
