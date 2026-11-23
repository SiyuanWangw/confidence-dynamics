#!/usr/bin/env python3
"""Score Section 4.2 / Table 1 from N=50 generation dumps.

Each dump is one model on one benchmark. Trajectories that hit the length cap
are already dropped. This script draws the paper's 20 windows of 30 original
sample slots, votes with the Table 1 rules, and prints accuracy in percent.

  python scripts/score_main_table.py
  python scripts/score_main_table.py --models Qwen3-8B --datasets hmmt aime
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from decimal import Decimal
from typing import Dict, List, Optional, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import orjson  # noqa: E402

from confdyn.table_vote import (  # noqa: E402
    COLUMN_KEYS,
    COLUMN_LABELS,
    DIRECT_KEYS,
    TOP50_KEYS,
    TOP80_KEYS,
    Trace,
    macro_average,
    mean_accuracy,
    round_half_up,
    window_starts,
)


DATASETS = (
    ("gpqa_diamond", "GPQA-D"),
    ("hmmt", "HMMT"),
    ("aime", "AIME"),
    ("sgpqa200", "SuperGPQA"),
    ("countdown6all", "Countdown6"),
)

# Directory name under models/, temperature, top-k. Countdown uses its own temperature.
MODELS = {
    "Qwen3-1.7B": {"dir": "Qwen3-1.7B", "temperature": "0.6", "top_k": "20", "countdown_temperature": "1.0"},
    "Qwen3-8B": {"dir": "Qwen3-8B", "temperature": "0.6", "top_k": "20", "countdown_temperature": "1.0"},
    "Nemotron-3-Nano-4B": {
        "dir": "NVIDIA-Nemotron-3-Nano-4B-BF16",
        "temperature": "1.0",
        "top_k": "-1",
        "countdown_temperature": "1.0",
    },
}


def _finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def traces_from_row(row: dict) -> List[Trace]:
    keys = list(row.get("extracted") or [])
    corrects = list(row.get("passat1_list") or [])
    kept = list(row.get("kept_indices") or [])
    n = len(keys)
    if n == 0 or len(corrects) != n:
        return []
    if len(kept) != n:
        kept = list(range(n))

    def column(name: str) -> List[Optional[float]]:
        values = list(row.get(name) or [])
        if len(values) != n:
            values = [None] * n
        return [_finite(value) for value in values]

    confidence = column("mean_last_c_list")
    tail_c = column("mean_tail_c_list")
    dynamics = column("mean_all_dc_list")
    tail_d = column("mean_tail_dc_list")
    return [
        Trace(
            slot=int(kept[j]),
            key=keys[j],
            correct=float(corrects[j]),
            confidence=confidence[j],
            tail_confidence=tail_c[j],
            dynamics=dynamics[j],
            tail_dynamics=tail_d[j],
        )
        for j in range(n)
    ]


def load_problems(path: str) -> List[List[Trace]]:
    with open(path, "rb") as handle:
        payload = orjson.loads(handle.read())
    runs = payload["runs"] if isinstance(payload.get("runs"), dict) else payload
    problems = []
    for row in runs.get("results") or []:
        problems.append(traces_from_row(row))
    if not problems:
        raise SystemExit(f"no problems in {path}")
    return problems


def find_dump(results: str, dataset: str, model: dict) -> str:
    """Plain N=50 dump: no seed, tree, or keep-truncated suffix."""
    temperature = model["countdown_temperature"] if dataset == "countdown6all" else model["temperature"]
    pattern = os.path.join(
        results,
        f"{dataset}_n1000",
        f"n50_{model['dir']}_t{temperature}_k{model['top_k']}_p0.95_mt*.json",
    )
    prefix = f"n50_{model['dir']}_t{temperature}_k{model['top_k']}_p0.95_mt"
    plain = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        token_budget = name[len(prefix):-5]
        if name.startswith(prefix) and name.endswith(".json") and token_budget.isdigit():
            plain.append(path)
    if len(plain) != 1:
        raise SystemExit(f"expected one N=50 dump, found {len(plain)} for {pattern}: {plain}")
    return plain[0]


def _cells(row: Dict[str, Decimal], keys: Sequence[str]) -> str:
    return " | ".join(str(round_half_up(row[key])) for key in keys)


def _header(keys: Sequence[str]) -> str:
    return " | ".join(COLUMN_LABELS[key] for key in keys)


def print_model(name: str, rows: List[tuple[str, Dict[str, Decimal]]]) -> None:
    overall = macro_average(row for _, row in rows)
    base = overall["standard"]
    print(f"\n## {name}\n")
    for title, keys in (
        ("Direct voting", DIRECT_KEYS),
        ("Top 80% by confidence, then vote", TOP80_KEYS),
        ("Top 50% by confidence, then vote", TOP50_KEYS),
    ):
        print(f"### {title}\n")
        print(f"| Dataset | {_header(keys)} |")
        print("|---|" + "---:|" * len(keys))
        for label, row in rows:
            print(f"| {label} | {_cells(row, keys)} |")
        print(f"| Overall | {_cells(overall, keys)} |")
        deltas = " | ".join(
            "—" if key == "standard" else _signed(overall[key] - base) for key in keys
        )
        print(f"| Δ | {deltas} |")
        print()


def _signed(value: Decimal) -> str:
    rounded = round_half_up(value)
    if rounded == 0:
        return "0.00"
    if rounded > 0:
        return f"+{rounded}"
    return str(rounded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=os.path.join(ROOT, "results"))
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--datasets", nargs="*", default=[name for name, _ in DATASETS])
    args = parser.parse_args()
    labels = dict(DATASETS)
    unknown = [name for name in args.models if name not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model: {unknown}. Choose from {list(MODELS)}")
    missing_ds = [name for name in args.datasets if name not in labels]
    if missing_ds:
        raise SystemExit(f"unknown dataset: {missing_ds}. Choose from {list(labels)}")

    starts = window_starts()
    print(
        "Table 1. Mean accuracy (%) over 20 windows of 30 traces drawn from each N=50 pool. "
        f"Window starts: {starts}."
    )
    print("CW / DW use the full trajectory. A T superscript uses the last 2048 tokens.")
    failed = False
    for model_name in args.models:
        spec = MODELS[model_name]
        rows = []
        for dataset in args.datasets:
            try:
                path = find_dump(args.results, dataset, spec)
            except SystemExit as exc:
                print(f"\n{model_name} {dataset}: {exc}", file=sys.stderr)
                failed = True
                continue
            problems = load_problems(path)
            rows.append((labels[dataset], mean_accuracy(problems, starts)))
            print(f"loaded {model_name} {dataset}: {len(problems)} problems  {path}", file=sys.stderr)
        if rows:
            print_model(model_name, rows)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
