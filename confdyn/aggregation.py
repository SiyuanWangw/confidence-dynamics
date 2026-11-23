"""Vote helpers used while a dump is written.

The numbers in Section 4.2 / Table 1 are not these per-dump fields. That table
draws 20 windows from an N=50 pool and applies the rules in
``confdyn.table_vote``. The joint C ∩ ΔC helpers below are an earlier filter
and stay available for offline inspection.

Baselines
---------
- Standard: unweighted majority vote over all N valid trajectories.
- Tail Conf. W.: weighted majority vote with weights = tail C.
- Tail Conf. F.@ρ: keep top ρ by tail C, then unweighted majority vote
  (ρ ∈ {0.20, 0.30, 0.50} by default).

Table 1 uses the last 2048 tokens for every model. ``resolve_tail_mode``
still returns a 20% tail for Gemma-named families; that path is not used by
``scripts/score_main_table.py``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import confdyn.filter_metrics as fm


# Paper defaults
DEFAULT_RHO_QWEN = 0.60
DEFAULT_RHO_GEMMA = 0.50
DEFAULT_TAIL_TOKENS_QWEN = 2048
DEFAULT_TAIL_FRAC_GEMMA = 0.20
BASELINE_FILTER_RHOS = (0.20, 0.30, 0.50)
# Always report Ours at both paper ρ defaults (Gemma 0.5 / Qwen 0.6).
JOINT_REPORT_RHOS = (0.50, 0.60)


def default_rho(family: str) -> float:
    fam = (family or "").lower()
    if fam.startswith("gemma"):
        return DEFAULT_RHO_GEMMA
    return DEFAULT_RHO_QWEN


def resolve_tail_mode(family: str) -> Tuple[str, float]:
    """Return (mode, value) where mode is 'tokens' or 'frac'."""
    fam = (family or "").lower()
    if fam.startswith("gemma"):
        return "frac", DEFAULT_TAIL_FRAC_GEMMA
    return "tokens", float(DEFAULT_TAIL_TOKENS_QWEN)


def trajectory_tail_mean(
    step_values: Optional[Sequence],
    family: str,
    *,
    tokens: Optional[int] = None,
    frac: Optional[float] = None,
) -> Optional[float]:
    """Mean of C or ΔC over the model-specific tail window."""
    mode, default_val = resolve_tail_mode(family)
    if mode == "frac":
        return fm.mean_tail_frac(step_values, frac if frac is not None else default_val)
    w = int(tokens if tokens is not None else default_val)
    return fm.mean_tail_steps(step_values, w)


def top_ratio_indices(
    scores: Sequence,
    rho: float,
    *,
    highest: bool = True,
) -> List[int]:
    """Indices of the top ρ proportion of scores (ceil). None scores sort last."""
    n = len(scores)
    if n == 0:
        return []
    rho = min(max(float(rho), 0.0), 1.0)
    if rho <= 0:
        return []
    k = max(1, int(math.ceil(n * rho)))
    return fm.topk_indices(scores, k, highest=highest)


def joint_filter_indices(
    conf_scores: Sequence,
    delta_scores: Sequence,
    rho: float,
) -> List[int]:
    """Intersection of Top_ρ(C) and Top_ρ(ΔC), preserving original index order."""
    if len(conf_scores) != len(delta_scores):
        raise ValueError("conf_scores and delta_scores must have the same length")
    keep_c = set(top_ratio_indices(conf_scores, rho, highest=True))
    keep_d = set(top_ratio_indices(delta_scores, rho, highest=True))
    return [i for i in range(len(conf_scores)) if i in keep_c and i in keep_d]


def _finite_or_none(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def best_score_index(scores: Sequence) -> Optional[int]:
    """Index of the highest finite score. Ties keep the earliest trajectory."""
    best_i = None
    best_s = None
    for i, score in enumerate(scores):
        value = _finite_or_none(score)
        if value is None:
            continue
        if best_s is None or value > best_s:
            best_s = value
            best_i = i
    return best_i


def _weighted_index(vote_keys: Sequence, scores: Sequence, majority_i: int) -> int:
    """Sum scores per answer. A non-positive pool falls back to majority."""
    weights = [_finite_or_zero(x) for x in scores]
    if all(w <= 0 for w in weights):
        return majority_i
    _, weighted_i = fm.weighted_majority_vote(vote_keys, weights)
    return weighted_i


def answer_selection(
    pass_list: Sequence[float],
    vote_keys: Sequence,
    tail_dc: Sequence,
    all_dc: Sequence,
) -> Dict[str, object]:
    """Five ways to turn the retained pool into one answer.

    Tail ΔC is the paper tail (Qwen: last 2048 steps). ΔC is the mean over
    every step of the trajectory. Best picks the single highest score. Vote
    counts extracted answers. Weighted sums the score inside each answer; a
    non-positive pool falls back to the unweighted majority. Ties keep the
    earliest trajectory.
    """
    n = len(pass_list)
    empty = {
        "vote_acc_best_tail_dc": 0.0,
        "best_tail_dc_index": None,
        "vote_acc_best_dc": 0.0,
        "best_dc_index": None,
        "vote_acc_majority": 0.0,
        "majority_index": None,
        "vote_acc_tail_dc_weighted": 0.0,
        "tail_dc_weighted_index": None,
        "vote_acc_dc_weighted": 0.0,
        "dc_weighted_index": None,
    }
    if n == 0:
        return empty
    if len(tail_dc) != n or len(all_dc) != n or len(vote_keys) != n:
        raise ValueError("pass_list, vote_keys, tail_dc, and all_dc must have the same length")
    best_tail_i = best_score_index(tail_dc)
    if best_tail_i is None:
        best_tail_i = 0
    best_dc_i = best_score_index(all_dc)
    if best_dc_i is None:
        best_dc_i = 0
    _, majority_i, _ = fm.majority_vote(vote_keys)
    tail_weighted_i = _weighted_index(vote_keys, tail_dc, majority_i)
    dc_weighted_i = _weighted_index(vote_keys, all_dc, majority_i)
    return {
        "vote_acc_best_tail_dc": float(pass_list[best_tail_i]),
        "best_tail_dc_index": best_tail_i,
        "vote_acc_best_dc": float(pass_list[best_dc_i]),
        "best_dc_index": best_dc_i,
        "vote_acc_majority": float(pass_list[majority_i]),
        "majority_index": majority_i,
        "vote_acc_tail_dc_weighted": float(pass_list[tail_weighted_i]),
        "tail_dc_weighted_index": tail_weighted_i,
        "vote_acc_dc_weighted": float(pass_list[dc_weighted_i]),
        "dc_weighted_index": dc_weighted_i,
    }


def _finite_or_zero(x) -> float:
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) else 0.0


def filter_then_majority_acc(
    pass_list: Sequence[float],
    vote_keys: Sequence,
    scores: Sequence,
    rho: float,
) -> float:
    """Keep top ρ by ``scores``, then unweighted majority vote."""
    idx = top_ratio_indices(scores, rho, highest=True)
    return fm.vote_on_subset(pass_list, vote_keys, idx)


def joint_delta_weighted_acc(
    pass_list: Sequence[float],
    vote_keys: Sequence,
    conf_scores: Sequence,
    delta_scores: Sequence,
    rho: float,
) -> Tuple[float, List[int]]:
    """Joint C@ρ ∩ ΔC@ρ filter, then ΔC-weighted majority vote.

    Returns (accuracy 0/1 for the voted trace, retained indices).
    Falls back to unweighted majority over the full pool if the intersection
    is empty (should be rare; Top_ρ sets always contain at least one index each).
    """
    retained = joint_filter_indices(conf_scores, delta_scores, rho)
    if not retained:
        voted, first_index, _ = fm.majority_vote(vote_keys)
        return float(pass_list[first_index]), list(range(len(vote_keys)))

    keys = [vote_keys[i] for i in retained]
    weights = [_finite_or_zero(delta_scores[i]) for i in retained]
    # If all ΔC weights are non-positive, fall back to unweighted majority on the set.
    if all(w <= 0 for w in weights):
        _, local_i, _ = fm.majority_vote(keys)
        chosen = retained[local_i]
        return float(pass_list[chosen]), retained

    _, local_i = fm.weighted_majority_vote(keys, weights)
    chosen = retained[local_i]
    return float(pass_list[chosen]), retained


def evaluate_paper_methods(
    pass_list: Sequence[float],
    vote_keys: Sequence,
    tail_c: Sequence,
    tail_dc: Sequence,
    *,
    rho: float,
    filter_rhos: Sequence[float] = BASELINE_FILTER_RHOS,
) -> Dict[str, object]:
    """Compute Standard / Tail-Conf baselines / Ours for one problem.

    Returns a dict of vote accuracies plus retained-set size for Ours.
    """
    n = len(pass_list)
    if n == 0:
        out: Dict[str, object] = {
            "vote_acc": 0.0,
            "vote_acc_tail_c_weighted": 0.0,
            "vote_acc_joint_dc_weighted": 0.0,
            "joint_retained": 0,
            "joint_rho": float(rho),
        }
        for r in filter_rhos:
            pct = int(round(100 * float(r)))
            out[f"vote_acc_tail_c_filter_{pct}"] = 0.0
            out[f"passat1_tail_c_filter_{pct}"] = None
        for jr in JOINT_REPORT_RHOS:
            pct = int(round(100 * float(jr)))
            out[f"vote_acc_joint_dc_weighted_{pct}"] = 0.0
            out[f"joint_retained_{pct}"] = 0
            out[f"joint_retained_indices_{pct}"] = []
            out[f"passat1_joint_{pct}"] = None
        return out

    voted, first_index, _ = fm.majority_vote(vote_keys)
    out = {
        "vote_acc": float(pass_list[first_index]),
        "vote_acc_tail_c_weighted": fm.weighted_vote_acc(
            pass_list, vote_keys, [_finite_or_zero(x) for x in tail_c]
        ),
        "joint_rho": float(rho),
    }
    for r in filter_rhos:
        pct = int(round(100 * float(r)))
        idx = top_ratio_indices(tail_c, r, highest=True)
        out[f"vote_acc_tail_c_filter_{pct}"] = filter_then_majority_acc(
            pass_list, vote_keys, tail_c, r
        )
        out[f"passat1_tail_c_filter_{pct}"] = (
            sum(float(pass_list[i]) for i in idx) / len(idx) if idx else None
        )

    joint_acc, retained = joint_delta_weighted_acc(
        pass_list, vote_keys, tail_c, tail_dc, rho
    )
    out["vote_acc_joint_dc_weighted"] = joint_acc
    out["joint_retained"] = len(retained)
    out["joint_retained_indices"] = retained
    for jr in JOINT_REPORT_RHOS:
        pct = int(round(100 * float(jr)))
        acc_r, retained_r = joint_delta_weighted_acc(
            pass_list, vote_keys, tail_c, tail_dc, jr
        )
        out[f"vote_acc_joint_dc_weighted_{pct}"] = acc_r
        out[f"joint_retained_{pct}"] = len(retained_r)
        out[f"joint_retained_indices_{pct}"] = retained_r
        out[f"passat1_joint_{pct}"] = (
            sum(float(pass_list[i]) for i in retained_r) / len(retained_r)
            if retained_r
            else None
        )
    return out
