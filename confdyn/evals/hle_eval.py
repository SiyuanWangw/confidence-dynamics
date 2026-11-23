"""Rule judge for the HLE 300-item short-answer subset (int / number / word).

Integer and other numeric answers reuse ``aime_eval`` (boxed extract +
``math_verify``), matching overgen-filter's ``gsm8k_eval`` path. English-word
answers use case-insensitive exact match after boxed / ``Answer:`` extract.
"""

from __future__ import annotations

import re

from confdyn.evals import aime_eval

_ANSWER_LINE = re.compile(r"(?im)^(?:final\s+)?answer(?:\s*choice)?\s*:\s*(.+)$")


def _norm_word(s: str) -> str:
    s = (s or "").strip()
    s = s.strip(" \t\r\n\"'`“”‘’.")
    return re.sub(r"\s+", " ", s).casefold()


def extract_word(pred_str: str) -> str:
    boxed = aime_eval.extract_answer(pred_str, use_last_number=False)
    if boxed:
        return boxed
    lines = _ANSWER_LINE.findall(pred_str or "")
    if lines:
        return lines[-1].strip().strip(".")
    return ""


def rule_judge(
    solution_str: str, sample: dict, finish_generation: bool = True
) -> tuple[bool, str]:
    gold = sample.get("answer")
    if gold is None:
        gold = sample.get("final_answer")
    gold = "" if gold is None else str(gold)
    kind = (sample.get("kind") or sample.get("source") or "").lower()
    if kind in ("word", "hle_word"):
        extracted = extract_word(solution_str)
        if not extracted:
            return False, "No extracted answer"
        return _norm_word(extracted) == _norm_word(gold), extracted
    return aime_eval.rule_judge(solution_str, gold, finish_generation)
