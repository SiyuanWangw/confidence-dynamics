#!/usr/bin/env python3
"""Build LiveCodeBench v6, contest dates 2025-02-01 through 2025-05-01.

release_v6 adds test6.jsonl (175 problems) on top of v5. The 25.02–25.05
window is the 131 of those whose contest_date is in
[2025-02-01, 2025-05-01], matching the official loader
(start <= contest_date <= end, both parsed as midnight).

Source file (kept compressed tests, as published):

  datasets/lcb_raw/test6.jsonl

Writes:

  datasets/lcb_v6.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "lcb_raw" / "test6.jsonl"
LABELED = ROOT / "datasets" / "lcb_raw" / "lcb_v6_labeled.jsonl"
OUT = ROOT / "datasets" / "lcb_v6.json"
START = datetime(2025, 2, 1)
END = datetime(2025, 5, 1)
EXPECTED = 131


def _in_window(stamp: str) -> bool:
    return START <= datetime.fromisoformat(stamp) <= END


def main() -> None:
    if not RAW.is_file():
        raise SystemExit(f"missing {RAW}")
    rows = []
    n_file = 0
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_file += 1
            row = json.loads(line)
            if _in_window(row["contest_date"]):
                rows.append(row)
    rows.sort(key=lambda r: (r["contest_date"], r["platform"], r["question_id"]))
    if LABELED.is_file():
        labeled = set()
        with LABELED.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if _in_window(row["contest_date"]):
                    labeled.add(row["question_id"])
        got = {r["question_id"] for r in rows}
        if got != labeled:
            missing = sorted(labeled - got)
            extra = sorted(got - labeled)
            raise SystemExit(
                f"window ids differ from the full v6 index: missing={missing[:5]} extra={extra[:5]}"
            )
    if len(rows) != EXPECTED:
        raise SystemExit(f"expected {EXPECTED} problems, found {len(rows)} (file has {n_file})")
    out = []
    for i, row in enumerate(rows):
        out.append(
            {
                "id": f"lcb_v6_{i:03d}",
                "question_id": row["question_id"],
                "question_title": row["question_title"],
                "question_content": row["question_content"],
                "platform": row["platform"],
                "contest_id": row["contest_id"],
                "contest_date": row["contest_date"],
                "difficulty": row["difficulty"],
                "starter_code": row.get("starter_code") or "",
                "public_test_cases": row["public_test_cases"],
                "private_test_cases": row["private_test_cases"],
                "metadata": row["metadata"],
            }
        )
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(out)} problems to {OUT} ({OUT.stat().st_size} bytes), source file {n_file}")


if __name__ == "__main__":
    main()
