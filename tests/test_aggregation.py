"""Tests for paper joint C@ρ ∩ ΔC@ρ + ΔC-weighted aggregation."""

import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from aggregation import (  # noqa: E402
    default_rho,
    evaluate_paper_methods,
    filter_then_majority_acc,
    joint_delta_weighted_acc,
    joint_filter_indices,
    resolve_tail_mode,
    top_ratio_indices,
    trajectory_tail_mean,
)


def test_top_ratio_ceil():
    scores = [0.1, 0.9, 0.2, 0.8, 0.3]  # n=5, ρ=0.4 → ceil(2)=2
    idx = top_ratio_indices(scores, 0.4)
    assert set(idx) == {1, 3}


def test_joint_filter_intersection():
    # C ranks: 4,3,2,1,0  → top 60% = {0,1,2}
    # ΔC ranks: 0,1,2,3,4 → top 60% = {4,3,2}
    # intersection = {2}
    c = [5, 4, 3, 2, 1]
    dc = [1, 2, 3, 4, 5]
    got = joint_filter_indices(c, dc, 0.6)
    assert got == [2]


def test_joint_delta_weighted_prefers_high_dc():
    # Retained by joint @50%: top-C={0,1}, top-ΔC={1,2} → {1}
    # With N=4 and ρ=0.5: top2 C = indices of 9,8 → {0,1}
    # top2 ΔC = indices of 10,9 → {1,2}; intersection {1}
    pass_list = [0.0, 1.0, 0.0, 0.0]
    keys = ["wrong", "right", "wrong", "wrong"]
    c = [9.0, 8.0, 1.0, 0.5]
    dc = [1.0, 10.0, 9.0, 0.1]
    acc, retained = joint_delta_weighted_acc(pass_list, keys, c, dc, 0.5)
    assert retained == [1]
    assert acc == 1.0


def test_joint_weighted_tiebreak_uses_dc_mass():
    # Intersection keeps two answers; higher ΔC mass wins.
    pass_list = [0.0, 1.0, 1.0, 0.0]
    keys = ["A", "B", "B", "A"]
    c = [10.0, 9.0, 8.0, 0.1]  # top 75% ≈ {0,1,2}
    dc = [1.0, 5.0, 4.0, 0.0]
    acc, retained = joint_delta_weighted_acc(pass_list, keys, c, dc, 0.75)
    assert set(retained) == {0, 1, 2}
    assert acc == 1.0  # B wins via ΔC weights 5+4 > 1


def test_tail_c_filter_baseline():
    pass_list = [0.0, 1.0, 1.0, 0.0]
    keys = ["x", "y", "y", "x"]
    c = [1.0, 9.0, 8.0, 2.0]
    # ρ=0.5 → keep {1,2} → majority y → correct
    assert filter_then_majority_acc(pass_list, keys, c, 0.5) == 1.0


def test_evaluate_paper_methods_keys():
    out = evaluate_paper_methods(
        [1.0, 0.0, 1.0, 0.0],
        ["a", "b", "a", "b"],
        [5.0, 1.0, 4.0, 2.0],
        [3.0, 0.5, 2.5, 0.2],
        rho=0.5,
    )
    assert out["vote_acc"] in (0.0, 1.0)
    assert "vote_acc_tail_c_weighted" in out
    assert "vote_acc_tail_c_filter_20" in out
    assert "vote_acc_tail_c_filter_30" in out
    assert "vote_acc_tail_c_filter_50" in out
    assert "vote_acc_joint_dc_weighted" in out
    assert out["joint_retained"] >= 1


def test_family_defaults():
    assert math.isclose(default_rho("qwen3"), 0.6)
    assert math.isclose(default_rho("gemma4"), 0.5)
    assert resolve_tail_mode("qwen3") == ("tokens", 2048.0)
    assert resolve_tail_mode("gemma4") == ("frac", 0.20)


def test_trajectory_tail_mean():
    xs = list(range(1, 11))  # 1..10
    assert math.isclose(trajectory_tail_mean(xs, "qwen3", tokens=3), 9.0)
    assert math.isclose(trajectory_tail_mean(xs, "gemma4"), 9.5)  # last 20% = 2 toks
