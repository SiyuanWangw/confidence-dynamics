#!/usr/bin/env python3
"""Recompute paper aggregation metrics from an existing generate.py dump.

Usage:
  python scripts/reaggregate.py results/hmmt.json --rho 0.5
  python scripts/reaggregate.py results/hmmt.json --rho 0.6 --family qwen3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import aggregation as agg  # noqa: E402


def _pick_lists(row: dict, family: str):
    if row.get("paper_tail_c_list") and row.get("paper_tail_dc_list"):
        return row["paper_tail_c_list"], row["paper_tail_dc_list"]
    mode, _ = agg.resolve_tail_mode(family)
    if mode == "frac":
        return row.get("mean_tail20_c_list") or [], row.get("mean_tail20_dc_list") or []
    return row.get("mean_tail_c_list") or [], row.get("mean_tail_dc_list") or []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dump_json")
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--family", default=None, help="qwen3 / gemma4 (for defaults)")
    args = p.parse_args()

    with open(args.dump_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    stats = payload.get("stats") or {}
    family = args.family or stats.get("family") or "qwen3"
    rho = float(args.rho) if args.rho is not None else agg.default_rho(family)

    rows = payload.get("results") or []
    keys = [
        "vote_acc",
        "vote_acc_tail_c_weighted",
        "vote_acc_tail_c_filter_20",
        "vote_acc_tail_c_filter_30",
        "vote_acc_tail_c_filter_50",
        "vote_acc_joint_dc_weighted",
    ]
    sums = {k: 0.0 for k in keys}
    n_ret = 0.0
    n = 0
    for row in rows:
        passes = row.get("passat1_list") or []
        if not passes:
            continue
        vote_keys = row.get("extracted")
        if not vote_keys or len(vote_keys) != len(passes):
            vote_keys = [bool(x >= 0.5) for x in passes]
        c, dc = _pick_lists(row, family)
        if len(c) != len(passes) or len(dc) != len(passes):
            continue
        out = agg.evaluate_paper_methods(passes, vote_keys, c, dc, rho=rho)
        for k in keys:
            sums[k] += float(out[k])
        n_ret += float(out["joint_retained"])
        n += 1

    if n == 0:
        print("no usable rows", file=sys.stderr)
        sys.exit(1)

    print(f"cases={n}  family={family}  rho={rho}")
    for k in keys:
        print(f"{k:32s} {100.0 * sums[k] / n:6.2f}%")
    print(f"{'mean_joint_retained':32s} {n_ret / n:6.2f}")


if __name__ == "__main__":
    main()
