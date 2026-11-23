"""CPU checks for countdown5 (use all 5) and countdown6 (4 of 6, 2+2)."""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from countdown_eval import rule_judge  # noqa: E402

COUNTDOWN5 = os.path.join(
    ROOT, "..", "Soft-Thinking", "datasets", "countdown5.jsonl"
)


def test_countdown5_gold_boxed():
    ok, extracted = rule_judge(
        r"think. \boxed{((((16 * 25) - 47) - 32) - 141)}",
        180,
        [25, 16, 32, 141, 47],
        variant="countdown5",
    )
    assert ok, extracted


def test_countdown5_rejects_missing_number():
    ok, _ = rule_judge(
        r"\boxed{(16 * 25) - 47}",
        353,
        [25, 16, 32, 141, 47],
        variant="countdown5",
    )
    assert not ok


def test_countdown6_gold_still_passes():
    ok, extracted = rule_judge(
        r"\boxed{(((165 - 161) * 22) + 157)}",
        245,
        [161, 96, 165, 157, 22, 29],
        variant="countdown6",
    )
    assert ok, extracted


def test_countdown5_file_golds():
    n = 0
    with open(COUNTDOWN5, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            boxed = r"\boxed{" + rec["answer"] + "}"
            ok, extracted = rule_judge(
                boxed, rec["target"], rec["nums"], variant="countdown5"
            )
            assert ok, (n, rec["answer"], rec["target"], extracted)
            n += 1
    assert n == 200
