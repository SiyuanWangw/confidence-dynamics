"""CPU checks for AIME 2026 loading and integer judging."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from generate import (
    DEFAULT_AIME26,
    MATH_QA_DATASETS,
    MATH_QUERY_TEMPLATE,
    _dataset_defaults,
    _judge_sample,
    _load_samples,
    _math_answer,
    _math_question,
    _user_prompt,
)
from gsm8k_eval import rule_judge


def test_aime26_is_math_qa():
    assert "aime26" in MATH_QA_DATASETS
    assert _dataset_defaults("aime26") == DEFAULT_AIME26


def test_aime26_has_thirty_problems():
    rows = _load_samples(DEFAULT_AIME26)
    assert len(rows) == 30
    sources = {r.get("source") for r in rows}
    assert sources == {"aime2026"}
    assert rows[0]["id"] == "aime2026_00"
    assert rows[-1]["id"] == "aime2026_29"
    assert _math_question(rows[0]).startswith("Patrick started walking")
    assert _math_answer(rows[0]) == "277"
    assert _math_answer(rows[1]) == "62"


def test_prompt_matches_aime_template():
    rows = _load_samples(DEFAULT_AIME26)
    prompt = _user_prompt("aime26", rows[0])
    assert prompt == MATH_QUERY_TEMPLATE.format(Question=_math_question(rows[0]))
    assert "\\boxed{}" in prompt


def test_original_aime_format():
    q = "Find the number of residents."
    sample = {
        "prompt": [{"from": "user", "value": q}],
        "final_answer": "277",
    }
    assert _math_question(sample) == q
    assert _math_answer(sample) == "277"


def test_judge_uses_gsm8k():
    sample = {"final_answer": "277"}
    ok, extracted = _judge_sample("aime26", r"Reasoning. \boxed{277}", sample, True)
    assert ok, extracted
    ok, _ = _judge_sample("aime26", r"\boxed{12}", sample, True)
    assert not ok


def test_boxed_exact_match():
    ok, extracted = rule_judge(r"Reasoning. \boxed{277}", "277")
    assert ok, extracted
