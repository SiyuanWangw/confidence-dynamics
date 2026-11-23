#!/usr/bin/env python3
"""Recompute the older joint-filter stats from a dump.

Table 1 (Section 4.2) is ``scripts/score_main_table.py``, not this script.
This entry point drops truncated traces and reprints the joint C ∩ ΔC fields.

Usage:
  python scripts/reaggregate.py results/hmmt.json
  python scripts/reaggregate.py results/hmmt.json --family qwen3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from confdyn import aggregation as agg  # noqa: E402
from confdyn.filter_metrics import mean_defined, untruncated_indices  # noqa: E402
from confdyn.pipeline import _STAT_PRINT_KEYS  # noqa: E402


def _pick_lists(row: dict, family: str):
    if row.get("paper_tail_c_list") and row.get("paper_tail_dc_list"):
        return row["paper_tail_c_list"], row["paper_tail_dc_list"]
    mode, _ = agg.resolve_tail_mode(family)
    if mode == "frac":
        return row.get("mean_tail20_c_list") or [], row.get("mean_tail20_dc_list") or []
    return row.get("mean_tail_c_list") or [], row.get("mean_tail_dc_list") or []


def _already_filtered(row: dict) -> bool:
    """New dumps store only the kept traces and set n_kept to that length."""
    passes = row.get("passat1_list") or []
    if "n_kept" not in row or not passes:
        return False
    return int(row.get("n_kept") or 0) == len(passes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dump_json")
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--family", default=None, help="qwen3 / gemma4 (for defaults)")
    args = p.parse_args()

    with open(args.dump_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    max_new = int((raw.get("args") or {}).get("max_new_tokens") or 0)
    payload = raw["runs"] if isinstance(raw.get("runs"), dict) else raw
    stats0 = payload.get("stats") or {}
    family = args.family or stats0.get("family") or "qwen3"
    rho = float(args.rho) if args.rho is not None else agg.default_rho(family)
    tail_mode = stats0.get("tail_mode") or agg.resolve_tail_mode(family)[0]

    sums = {}
    c_ok, c_bad, dc_ok, dc_bad = [], [], [], []
    tok_sum = 0
    tok_n = 0
    n_cases = 0
    n_drop = 0
    for row in payload.get("results") or []:
        passes = list(row.get("passat1_list") or [])
        if not passes:
            continue
        vote_keys = row.get("extracted")
        if not vote_keys or len(vote_keys) != len(passes):
            vote_keys = [bool(x >= 0.5) for x in passes]
        c, dc = _pick_lists(row, family)
        if len(c) != len(passes) or len(dc) != len(passes):
            continue
        toks = list(row.get("generated_tokens") or [])
        if _already_filtered(row):
            idx = list(range(len(passes)))
            n_drop += int(row.get("n_truncated") or 0)
        else:
            fins = row.get("finished")
            if not isinstance(fins, list) or len(fins) != len(passes):
                fins = [None] * len(passes)
            idx = untruncated_indices(fins, toks, max_new)
        n_drop += len(passes) - len(idx)
        if not idx:
            continue
        passes = [passes[i] for i in idx]
        vote_keys = [vote_keys[i] for i in idx]
        c = [c[i] for i in idx]
        dc = [dc[i] for i in idx]
        out = agg.evaluate_paper_methods(passes, vote_keys, c, dc, rho=rho)
        rec = {
            "pass@1": sum(passes) / len(passes),
            "pass@k": max(passes),
            "vote_acc": out["vote_acc"],
            "vote_acc_tail_c_weighted": out["vote_acc_tail_c_weighted"],
            "vote_acc_tail_c_filter_30": out["vote_acc_tail_c_filter_30"],
            "pass@1_tail_c_filter_30": out["passat1_tail_c_filter_30"],
            "vote_acc_tail_c_filter_50": out["vote_acc_tail_c_filter_50"],
            "pass@1_tail_c_filter_50": out["passat1_tail_c_filter_50"],
            "vote_acc_joint_dc_weighted_50": out["vote_acc_joint_dc_weighted_50"],
            "vote_acc_joint_dc_weighted_60": out["vote_acc_joint_dc_weighted_60"],
            "pass@1_joint_50": out["passat1_joint_50"],
            "pass@1_joint_60": out["passat1_joint_60"],
            "mean_joint_retained_50": out["joint_retained_50"],
            "mean_joint_retained_60": out["joint_retained_60"],
        }
        for k, v in rec.items():
            sums.setdefault(k, []).append(v)
        for i, passed in enumerate(passes):
            if c[i] is None:
                continue
            (c_ok if float(passed) >= 0.5 else c_bad).append(float(c[i]))
            if dc[i] is not None:
                (dc_ok if float(passed) >= 0.5 else dc_bad).append(float(dc[i]))
        for i in idx:
            if i < len(toks) and toks[i] is not None:
                tok_sum += int(toks[i])
                tok_n += 1
        n_cases += 1

    if n_cases == 0:
        print("no usable rows", file=sys.stderr)
        sys.exit(1)

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    stats = {
        "total_num": n_cases,
        "pass@1": avg(sums["pass@1"]),
        "pass@k": avg(sums["pass@k"]),
        "vote_acc": avg(sums["vote_acc"]),
        "vote_acc_tail_c_weighted": avg(sums["vote_acc_tail_c_weighted"]),
        "vote_acc_tail_c_filter_30": avg(sums["vote_acc_tail_c_filter_30"]),
        "pass@1_tail_c_filter_30": avg(sums["pass@1_tail_c_filter_30"]),
        "vote_acc_tail_c_filter_50": avg(sums["vote_acc_tail_c_filter_50"]),
        "pass@1_tail_c_filter_50": avg(sums["pass@1_tail_c_filter_50"]),
        "vote_acc_joint_dc_weighted_50": avg(sums["vote_acc_joint_dc_weighted_50"]),
        "vote_acc_joint_dc_weighted_60": avg(sums["vote_acc_joint_dc_weighted_60"]),
        "pass@1_joint_50": avg(sums["pass@1_joint_50"]),
        "pass@1_joint_60": avg(sums["pass@1_joint_60"]),
        "mean_joint_retained_50": avg(sums["mean_joint_retained_50"]),
        "mean_joint_retained_60": avg(sums["mean_joint_retained_60"]),
        "joint_rho": rho,
        "tail_mode": tail_mode,
        "mean_tail_c_correct": mean_defined(c_ok),
        "mean_tail_c_incorrect": mean_defined(c_bad),
        "mean_tail_dc_correct": mean_defined(dc_ok),
        "mean_tail_dc_incorrect": mean_defined(dc_bad),
        "avg_token_length-all": (tok_sum / tok_n) if tok_n else None,
        "mean_unique_seq_ratio": None,
        "generate_s": stats0.get("generate_s"),
        "judge_s": stats0.get("judge_s"),
        "wall_s": stats0.get("wall_s"),
    }
    dataset = stats0.get("dataset") or "dump"
    print(f"\n=== {dataset} statistics (truncated excluded, dropped={n_drop}) ===")
    for key in _STAT_PRINT_KEYS:
        val = stats.get(key)
        if val is None:
            print(f"  {key}: n/a")
        elif isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
