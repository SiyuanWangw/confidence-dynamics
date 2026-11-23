"""CPU checks for merged AIME24+25 loading and integer judging."""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from generate import DEFAULT_AIME, MATH_QA_DATASETS, _load_samples, _math_answer, _math_question
from gsm8k_eval import rule_judge


def test_aime_is_math_qa():
    assert "aime" in MATH_QA_DATASETS


def test_merged_aime_has_both_years():
    rows = _load_samples(DEFAULT_AIME)
    assert len(rows) == 60
    sources = {r["source"] for r in rows}
    assert sources == {"aime2024", "aime2025"}
    assert _math_question(rows[0])
    assert _math_answer(rows[0]) == "73"
    assert _math_answer(rows[30]) == "70"


def test_original_aime_format():
    q = "Find the number of residents."
    sample = {
        "prompt": [{"from": "user", "value": q}],
        "final_answer": "73",
    }
    assert _math_question(sample) == q
    assert _math_answer(sample) == "73"


def test_boxed_exact_match():
    ok, extracted = rule_judge(r"Reasoning. \boxed{73}", "73")
    assert ok, extracted


def test_wrong_answer():
    ok, _ = rule_judge(r"\boxed{12}", "73")
    assert not ok


def test_no_box_falls_back_to_last_number():
    ok, extracted = rule_judge("The answer is 73", "73")
    assert ok, extracted
