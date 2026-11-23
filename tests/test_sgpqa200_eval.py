"""CPU checks for SuperGPQA PCB-hard 200-item subset."""

import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from generate import (
    DEFAULT_SGPQA200,
    GPQA_QUERY_TEMPLATE,
    MCQ_DATASETS,
    _dataset_defaults,
    _judge_sample,
    _load_samples,
    _user_prompt,
)
from gpqa_eval import choice_letters, extract_choice, format_gpqa_question, rule_judge


def test_sgpqa200_is_mcq():
    assert "sgpqa200" in MCQ_DATASETS
    assert _dataset_defaults("sgpqa200") == DEFAULT_SGPQA200


def test_sgpqa200_has_two_hundred_hard_pcb():
    rows = _load_samples(DEFAULT_SGPQA200)
    assert len(rows) == 200
    assert rows[0]["id"] == "sgpqa200_000"
    assert rows[-1]["id"] == "sgpqa200_199"
    assert all(r["difficulty"] == "hard" for r in rows)
    fields = Counter(r["field"] for r in rows)
    assert fields == {"Physics": 110, "Chemistry": 70, "Biology": 20}
    assert len({r["uuid"] for r in rows}) == 200
    assert all(r["final_answer"] in r["choices"] for r in rows)
    assert all(4 <= r["n_options"] <= 10 for r in rows)
    assert all(len(r["choices"]) == r["n_options"] for r in rows)


def test_prompt_uses_actual_letters():
    rows = _load_samples(DEFAULT_SGPQA200)
    ten = max(rows, key=lambda r: r["n_options"])
    letters = choice_letters(ten)
    assert letters == "ABCDEFGHIJ"
    prompt = _user_prompt("sgpqa200", ten)
    assert "one of A, B, C, D, E, F, G, H, I, J" in prompt
    q = format_gpqa_question(ten)
    assert q in prompt
    assert "J) " in q
    assert "ANSWER: [LETTER]" in prompt
    assert GPQA_QUERY_TEMPLATE.startswith("Answer the following multiple choice question.")


def test_extracts_j():
    sample = {
        "choices": {let: f"opt {let}" for let in "ABCDEFGHIJ"},
        "final_answer": "J",
        "n_options": 10,
    }
    ok, extracted = _judge_sample("sgpqa200", "lots of A in the stem.\nANSWER: J", sample, True)
    assert ok
    assert extracted == "J"
    ok, extracted = rule_judge(r"done \boxed{J}", "J", letters="ABCDEFGHIJ")
    assert ok
    assert extracted == "J"
    assert extract_choice("ANSWER: J") is None
    assert extract_choice("ANSWER: J", letters="ABCDEFGHIJ") == "J"


def test_wrong_letter():
    sample = {
        "choices": {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"},
        "final_answer": "E",
    }
    ok, extracted = _judge_sample("sgpqa200", "ANSWER: C", sample, True)
    assert not ok
    assert extracted == "C"
