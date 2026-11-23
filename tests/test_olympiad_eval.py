"""CPU checks for OlympiadBench boxed-answer judging."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from olympiad_eval import rule_judge  # noqa: E402


def test_boxed_exact_match():
    ok, extracted = rule_judge(r"Reasoning. \boxed{2}", "2")
    assert ok, extracted


def test_boxed_latex_match():
    ok, extracted = rule_judge(r"done \boxed{\frac{1}{2n+2}}", r"$\frac{1}{2n+2}$")
    assert ok, extracted


def test_wrong_answer():
    ok, _ = rule_judge(r"\boxed{3}", "2")
    assert not ok


def test_no_box():
    ok, extracted = rule_judge("I am not sure.", "2")
    assert not ok
    assert extracted == "No extracted answer"
