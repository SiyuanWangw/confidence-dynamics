"""CPU checks for the 300-item HLE short-answer subset."""

import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from confdyn.evals import hle_eval
from confdyn.generate import (
    DEFAULT_HLE300,
    MATH_QA_DATASETS,
    MATH_QUERY_TEMPLATE,
    _dataset_defaults,
    _judge_sample,
    _load_samples,
    _math_answer,
    _math_question,
    _user_prompt,
)


def test_hle300_is_math_qa():
    assert "hle300" in MATH_QA_DATASETS
    assert _dataset_defaults("hle300") == DEFAULT_HLE300


def test_hle300_has_three_hundred():
    rows = _load_samples(DEFAULT_HLE300)
    assert len(rows) == 300
    kinds = Counter(r["kind"] for r in rows)
    assert kinds == {"int": 100, "number": 100, "word": 100}
    assert rows[0]["id"] == "hle300_000"
    assert rows[-1]["id"] == "hle300_299"
    assert all(r.get("hle_id") for r in rows)
    assert all(_math_question(r) for r in rows)
    assert all(_math_answer(r) for r in rows)


def test_prompt_matches_math_template():
    rows = _load_samples(DEFAULT_HLE300)
    prompt = _user_prompt("hle300", rows[0])
    assert prompt == MATH_QUERY_TEMPLATE.format(Question=_math_question(rows[0]))
    assert "\\boxed{}" in prompt


def test_int_uses_numeric_judge():
    sample = {"kind": "int", "final_answer": "12"}
    ok, extracted = _judge_sample("hle300", r"Reasoning. \boxed{12}", sample, True)
    assert ok, extracted
    ok, _ = _judge_sample("hle300", r"\boxed{11}", sample, True)
    assert not ok


def test_number_uses_numeric_judge():
    sample = {"kind": "number", "answer": "0.5"}
    ok, extracted = hle_eval.rule_judge(r"done \boxed{0.5}", sample)
    assert ok, extracted
    ok, _ = hle_eval.rule_judge(r"\boxed{0.6}", sample)
    assert not ok


def test_word_is_case_insensitive():
    sample = {"kind": "word", "final_answer": "Helium"}
    ok, extracted = _judge_sample("hle300", r"The element is \boxed{helium}", sample, True)
    assert ok, extracted
    ok, _ = _judge_sample("hle300", r"\boxed{hydrogen}", sample, True)
    assert not ok
