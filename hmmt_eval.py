"""HMMT Feb+Nov 2025 rule judge.

Gold answers are LaTeX (fractions, radicals, mixed expressions), so this uses
the same boxed / latex2sympy verify path as OlympiadBench rather than GSM8K's
integer extractor.
"""

from __future__ import annotations

from olympiad_eval import rule_judge

__all__ = ["rule_judge"]
