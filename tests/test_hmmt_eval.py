"""CPU checks for merged HMMT Feb+Nov 2025 loading and latex judging."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from confdyn.generate import (
    DEFAULT_HMMT,
    MATH_QA_DATASETS,
    MATH_QUERY_TEMPLATE,
    _load_samples,
    _math_answer,
    _math_question,
    _user_prompt,
)
from confdyn.evals.hmmt_eval import rule_judge


def test_hmmt_is_math_qa():
    assert "hmmt" in MATH_QA_DATASETS


def test_merged_hmmt_has_both_contests():
    rows = _load_samples(DEFAULT_HMMT)
    assert len(rows) == 60
    sources = {r["source"] for r in rows}
    assert sources == {"hmmt_feb_2025", "hmmt_nov_2025"}
    assert _math_question(rows[0])
    assert _math_answer(rows[0]) == "103"
    assert _math_answer(rows[30]) == "3"


def test_prompt_matches_math_template():
    sample = {
        "question": "Compute $1+1$.",
        "answer": "2",
        "prompt": [{"from": "user", "value": "Compute $1+1$."}],
        "final_answer": "2",
    }
    prompt = _user_prompt("hmmt", sample)
    assert prompt == MATH_QUERY_TEMPLATE.format(Question="Compute $1+1$.")
    assert "\\boxed{}" in prompt


def test_prompt_list_format():
    q = "Compute the sum of the positive divisors of $9!$ that have units digit $1$."
    sample = {
        "prompt": [{"from": "user", "value": q}],
        "final_answer": "103",
    }
    assert _math_question(sample) == q
    assert _math_answer(sample) == "103"


def test_boxed_integer():
    ok, extracted = rule_judge(r"Reasoning. \boxed{103}", "103")
    assert ok, extracted


def test_boxed_latex_frac():
    ok, extracted = rule_judge(r"done \boxed{\frac{1}{576}}", r"\frac{1}{576}")
    assert ok, extracted


def test_boxed_slash_frac():
    ok, extracted = rule_judge(r"\boxed{1/91}", "1/91")
    assert ok, extracted


def test_wrong_answer():
    ok, _ = rule_judge(r"\boxed{12}", "103")
    assert not ok
