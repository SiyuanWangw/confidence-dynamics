"""CPU checks for countdown6 (4 of 6) and countdown6all (use all 6)."""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from confdyn.evals.countdown_eval import rule_judge  # noqa: E402


def test_countdown6_gold_still_passes():
    ok, extracted = rule_judge(
        r"\boxed{(((165 - 161) * 22) + 157)}",
        245,
        [161, 96, 165, 157, 22, 29],
        variant="countdown6",
    )
    assert ok, extracted


def test_countdown6_rejects_wrong_partition():
    # Uses 3 from the first half → invalid for countdown6.
    ok, _ = rule_judge(
        r"\boxed{(((161 + 96) + 165) - 157)}",
        265,
        [161, 96, 165, 157, 22, 29],
        variant="countdown6",
    )
    assert not ok


def test_countdown6all_file_golds():
    path = os.path.join(ROOT, "datasets", "countdown6_all.jsonl")
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            boxed = r"\boxed{" + rec["answer"] + "}"
            ok, extracted = rule_judge(
                boxed, rec["target"], rec["nums"], variant="countdown6all"
            )
            assert ok, (n, rec["answer"], rec["target"], extracted)
            n += 1
    assert n == 200
