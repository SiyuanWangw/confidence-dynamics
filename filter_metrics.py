"""Over-generate then filter + majority-vote helpers.

Generate N traces, score them, then either:
  - keep the k highest/lowest by a metric and report pass@1 / pass@k, or
  - majority-vote extracted answers (plain, filtered, or weighted).
"""

from __future__ import annotations

import math
import os
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple


def local_tail_window(default: int = 2048) -> int:
    """DeepConf local/tail window: mean over the last W decode steps."""
    raw = os.environ.get("LOCAL_TAIL_WINDOW", str(default))
    try:
        return max(int(raw), 1)
    except (TypeError, ValueError):
        return default


def majority_vote(keys: Sequence) -> Tuple[object, int, Counter]:
    """Most common key; first index among ties (stable, matches angular-sampling)."""
    if not keys:
        raise ValueError("majority_vote requires at least one key")
    counts = Counter(keys)
    voted = counts.most_common(1)[0][0]
    first_index = list(keys).index(voted)
    return voted, first_index, counts


def weighted_majority_vote(keys: Sequence, weights: Sequence[float]) -> Tuple[object, int]:
    """Sum weights per key; ties break by first occurrence of the winning key."""
    if not keys:
        raise ValueError("weighted_majority_vote requires at least one key")
    if len(weights) != len(keys):
        raise ValueError("keys and weights must have the same length")
    totals: dict = {}
    first: dict = {}
    for i, (key, w) in enumerate(zip(keys, weights)):
        ww = 0.0 if w is None or not math.isfinite(float(w)) else float(w)
        totals[key] = totals.get(key, 0.0) + ww
        first.setdefault(key, i)
    voted = max(totals, key=lambda k: (totals[k], -first[k]))
    return voted, first[voted]


def split_by_score(
    pass_list: Sequence[float],
    scores: Sequence,
    k: int,
    highest: bool = True,
) -> Tuple[float, float, float, float]:
    """k highest (or lowest) by `scores` vs the complement.

    None scores sort to the unused end.
    Returns (pass@1_sel, pass@k_sel, pass@1_rest, pass@k_rest).
    """
    n = len(pass_list)
    if n == 0 or k <= 0:
        return 0.0, 0.0, 0.0, 0.0
    k = min(int(k), n)

    def key(i):
        s = scores[i] if i < len(scores) else None
        if s is None:
            return float("-inf") if highest else float("inf")
        return float(s)

    order = sorted(range(n), key=key, reverse=highest)
    sel, rest = order[:k], order[k:]

    def p1(idx):
        return sum(pass_list[i] for i in idx) / len(idx) if idx else 0.0

    def pk(idx):
        return max(pass_list[i] for i in idx) if idx else 0.0

    return p1(sel), pk(sel), p1(rest), pk(rest)


def subset_pass1(
    pass_list: Sequence[float], scores: Sequence[float], k: int, highest: bool
) -> float:
    """Mean correctness of the k samples with highest/lowest `scores`."""
    n = len(pass_list)
    if n == 0 or k <= 0:
        return 0.0
    k = min(int(k), n)
    order = sorted(range(n), key=lambda i: scores[i], reverse=highest)
    picked = order[:k]
    return sum(pass_list[i] for i in picked) / k


def topk_indices(scores: Sequence, k: int, highest: bool = True) -> List[int]:
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    k = min(int(k), n)

    def key(i):
        s = scores[i]
        if s is None:
            return float("-inf") if highest else float("inf")
        return float(s)

    return sorted(range(n), key=key, reverse=highest)[:k]


def vote_on_subset(
    pass_list: Sequence[float],
    vote_keys: Sequence,
    indices: Sequence[int],
) -> float:
    """Majority-vote accuracy on a subset of traces (0/1 for the voted trace)."""
    if not indices:
        return 0.0
    keys = [vote_keys[i] for i in indices]
    _, local_i, _ = majority_vote(keys)
    chosen = indices[local_i]
    return float(pass_list[chosen])


def weighted_vote_acc(
    pass_list: Sequence[float],
    vote_keys: Sequence,
    weights: Sequence[float],
) -> float:
    _, idx = weighted_majority_vote(vote_keys, weights)
    return float(pass_list[idx])


def confidence_weights(mean_logprobs: Sequence, ents: Sequence) -> List[float]:
    """Prefer high mean logprob; fall back to inverse entropy."""
    weights = []
    for lp, ent in zip(mean_logprobs, ents):
        if lp is not None and math.isfinite(float(lp)):
            weights.append(math.exp(float(lp)))
            continue
        e = 0.0 if ent is None else max(float(ent), 0.0)
        weights.append(1.0 / (1.0 + e))
    return weights


def mean_defined(vals: Iterable) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def mean_steps(xs: Optional[Sequence]) -> Optional[float]:
    """Mean of a per-token customized_info list."""
    if not xs:
        return None
    vals = []
    for x in xs:
        if x is None:
            continue
        try:
            vals.append(float(x))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def defined_floats(xs: Optional[Sequence]) -> List[float]:
    out: List[float] = []
    for x in xs or []:
        if x is None:
            continue
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            continue
    return out


def mean_of_tail(vals: Sequence[float], n_tail: int) -> Optional[float]:
    if not vals:
        return None
    tail = vals[-max(int(n_tail), 1) :]
    return sum(tail) / len(tail)


def tail_frac_n(n: int, frac: float) -> int:
    if n <= 0:
        return 1
    return max(1, int(math.ceil(n * float(frac))))


def mean_tail_steps(
    xs: Optional[Sequence],
    window: Optional[int] = None,
) -> Optional[float]:
    """Mean of the last ``window`` defined steps (DeepConf local/tail)."""
    vals = defined_floats(xs)
    w = local_tail_window() if window is None else max(int(window), 1)
    return mean_of_tail(vals, w)


def mean_tail_frac(xs: Optional[Sequence], frac: float) -> Optional[float]:
    """Mean of the last ``frac`` of defined steps (e.g. 0.10 / 0.20)."""
    vals = defined_floats(xs)
    return mean_of_tail(vals, tail_frac_n(len(vals), frac))


def mean_steps_where(
    vals: Optional[Sequence],
    mask: Optional[Sequence],
    thresh: float = 0.5,
    invert: bool = False,
) -> Optional[float]:
    """Mean of ``vals`` on steps where ``mask >= thresh`` (or ``< thresh`` if invert).

    invert=True is the switch-step complement of penult_match (last vs penult top-1).
    """
    if not vals:
        return None
    m = mask or []
    kept = []
    for i, v in enumerate(vals):
        if v is None:
            continue
        flag = m[i] if i < len(m) else None
        if flag is None:
            continue
        keep = float(flag) >= thresh
        if invert:
            keep = not keep
        if not keep:
            continue
        try:
            kept.append(float(v))
        except (TypeError, ValueError):
            continue
    if not kept:
        return None
    return sum(kept) / len(kept)


def avg_or_none(rows: Sequence[dict], key: str) -> Optional[float]:
    return mean_defined(r.get(key) for r in rows)


def step_values(xs: Optional[Sequence], ndigits: int = 6) -> List[float]:
    """Drop None pads and round per-decode-step customized_info lists."""
    out: List[float] = []
    for x in xs or []:
        if x is None:
            continue
        try:
            out.append(round(float(x), ndigits))
        except (TypeError, ValueError):
            continue
    return out


def step_match_flags(xs: Optional[Sequence], thresh: float = 0.5) -> List[int]:
    """0/1 last-vs-penult top-1 match flags, dropping None pads."""
    out: List[int] = []
    for x in xs or []:
        if x is None:
            continue
        try:
            out.append(1 if float(x) >= thresh else 0)
        except (TypeError, ValueError):
            continue
    return out


def avg_select_k(
    results: Sequence[dict],
    score_list_key: str,
    k: int,
    highest: bool = True,
    pass_list_key: str = "passat1_list",
) -> Optional[float]:
    """Mean pass@1 of selecting ``k`` traces by ``score_list_key`` on each case."""
    accs = []
    for r in results:
        scores = r.get(score_list_key)
        passes = r.get(pass_list_key)
        if not scores or not passes:
            continue
        p1, _, _, _ = split_by_score(passes, scores, k, highest)
        accs.append(p1)
    return mean_defined(accs)
