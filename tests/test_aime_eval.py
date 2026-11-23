"""CPU checks for merged AIME 2024+2025+2026 loading and integer judging."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from confdyn.evals.aime_eval import rule_judge
from confdyn.generate import (
    DEFAULT_AIME,
    MATH_QA_DATASETS,
    MATH_QUERY_TEMPLATE,
    _dataset_defaults,
    _judge_sample,
    _load_samples,
    _math_answer,
    _math_question,
    _user_prompt,
)


def test_aime_is_math_qa():
    assert "aime" in MATH_QA_DATASETS
    assert "aime26" not in MATH_QA_DATASETS
    assert _dataset_defaults("aime") == DEFAULT_AIME


def test_merged_aime_has_ninety_problems():
    rows = _load_samples(DEFAULT_AIME)
    assert len(rows) == 90
    assert sum(r["source"] == "aime2024" for r in rows) == 30
    assert sum(r["source"] == "aime2025" for r in rows) == 30
    assert sum(r["source"] == "aime2026" for r in rows) == 30
    assert rows[0]["id"] == "aime2024_00"
    assert rows[29]["id"] == "aime2024_29"
    assert rows[30]["id"] == "aime2025_00"
    assert rows[59]["id"] == "aime2025_29"
    assert rows[60]["id"] == "aime2026_00"
    assert rows[-1]["id"] == "aime2026_29"
    assert len({r["id"] for r in rows}) == 90
    assert _math_question(rows[0]).startswith("Among the $900$ residents")
    assert _math_answer(rows[0]) == "73"
    assert _math_answer(rows[30]) == "70"
    assert _math_answer(rows[59]) == "240"
    assert _math_question(rows[60]).startswith("Patrick started walking")
    assert _math_answer(rows[60]) == "277"
    assert _math_answer(rows[61]) == "62"
    assert _math_answer(rows[-1]) == "393"


def test_prompt_matches_math_template():
    rows = _load_samples(DEFAULT_AIME)
    prompt = _user_prompt("aime", rows[60])
    assert prompt == MATH_QUERY_TEMPLATE.format(Question=_math_question(rows[60]))
    assert "\\boxed{}" in prompt


def test_original_prompt_list_format():
    q = "Find the number of residents."
    sample = {
        "prompt": [{"from": "user", "value": q}],
        "final_answer": "277",
    }
    assert _math_question(sample) == q
    assert _math_answer(sample) == "277"


def test_judge_uses_integer_extractor():
    sample = {"final_answer": "277"}
    ok, extracted = _judge_sample("aime", r"Reasoning. \boxed{277}", sample, True)
    assert ok, extracted
    ok, _ = _judge_sample("aime", r"\boxed{12}", sample, True)
    assert not ok
    ok, extracted = _judge_sample("aime", r"done \boxed{73}", {"answer": "73"}, True)
    assert ok, extracted


def test_boxed_exact_match():
    ok, extracted = rule_judge(r"Reasoning. \boxed{277}", "277")
    assert ok, extracted
