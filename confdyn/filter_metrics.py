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


def untruncated_indices(
    finished: Sequence,
    token_counts: Sequence,
    max_new_tokens: int,
) -> List[int]:
    """Indices of traces that finished before the length cap.

    Drop a trace when ``finished`` is False (finish reason is not stop) or when
    the completion length reached ``max_new_tokens``. Unknown finish with a
    shorter completion is kept.
    """
    n = max(len(finished), len(token_counts))
    cap = int(max_new_tokens or 0)
    keep: List[int] = []
    for i in range(n):
        fin = finished[i] if i < len(finished) else None
        tok = token_counts[i] if i < len(token_counts) else None
        if fin is False:
            continue
        if cap > 0 and tok is not None and int(tok) >= cap:
            continue
        keep.append(i)
    return keep


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


def step_layer_confidences(meta: dict, depth: int = 6) -> List[List[Optional[float]]]:
    """Per decode step, DeepConf C from the older layer to the last layer.

    Inner order is ``[C_{L-depth+1}, ..., C_{L-1}, C_L]``. A step is kept when
    its last-layer value is present, matching ``step_values`` on ``last_c``.
    """
    last = list(meta.get("last_c") or [])
    earlier = []
    for offset in range(max(int(depth), 1) - 1, 0, -1):
        earlier.append(list(meta.get(f"layer_c_{offset}") or []))
    series = earlier + [last]
    n = max((len(col) for col in series), default=0)
    rows: List[List[Optional[float]]] = []
    for i in range(n):
        last_v = last[i] if i < len(last) else None
        if last_v is None:
            continue
        row: List[Optional[float]] = []
        for col in series:
            value = col[i] if i < len(col) else None
            if value is None:
                row.append(None)
                continue
            try:
                row.append(round(float(value), 6))
            except (TypeError, ValueError):
                row.append(None)
        rows.append(row)
    return rows


def _id_overlap(earlier, last) -> Optional[float]:
    if not earlier or not last:
        return None
    try:
        last_ids = [int(x) for x in last]
        earlier_ids = {int(x) for x in earlier}
    except (TypeError, ValueError):
        return None
    if not last_ids:
        return None
    hit = sum(1 for x in last_ids if x in earlier_ids)
    return hit / len(last_ids)


def step_layer_top_overlap(meta: dict, depth: int = 6):
    """Per decode step, own top-k ids and overlap with the last layer.

    Inner order matches ``step_layer_confidences``: older layer first, last
    layer last. Overlap of the last column is 1. A step is kept when ``last_c``
    at that index is present. Overlap is ``|top(layer) ∩ top(L)| / |top(L)|``.
    """
    depth = max(int(depth), 1)
    last_c = list(meta.get("last_c") or [])
    columns = []
    for offset in range(depth - 1, -1, -1):
        columns.append(list(meta.get(f"layer_top_{offset}") or []))
    n = len(last_c)
    top_rows: List[List[Optional[List[int]]]] = []
    overlap_rows: List[List[Optional[float]]] = []
    for i in range(n):
        if last_c[i] is None:
            continue
        ids_row = []
        for col in columns:
            raw = col[i] if i < len(col) else None
            if not raw:
                ids_row.append(None)
                continue
            try:
                ids_row.append([int(x) for x in raw])
            except (TypeError, ValueError):
                ids_row.append(None)
        last_ids = ids_row[-1] if ids_row else None
        overlap = []
        for ids in ids_row:
            rate = _id_overlap(ids, last_ids)
            overlap.append(None if rate is None else round(rate, 6))
        overlap_rows.append(overlap)
        top_rows.append(ids_row)
    return top_rows, overlap_rows


def step_layer_top_vectors(meta: dict, prefix: str, depth: int = 6):
    """Per-step vectors aligned with ``step_layer_top_overlap``.

    ``prefix`` is ``layer_top_logit_``, ``layer_top_prob_``, or ``layer_top_cos_``.
    Column order is older layer first, last layer last. A step is kept when
    ``last_c`` at that index is present.
    """
    depth = max(int(depth), 1)
    last_c = list(meta.get("last_c") or [])
    columns = []
    for offset in range(depth - 1, -1, -1):
        columns.append(list(meta.get(f"{prefix}{offset}") or []))
    rows: List[List[Optional[List[float]]]] = []
    for i, c in enumerate(last_c):
        if c is None:
            continue
        row: List[Optional[List[float]]] = []
        for col in columns:
            raw = col[i] if i < len(col) else None
            if not raw:
                row.append(None)
                continue
            try:
                row.append([float(x) for x in raw])
            except (TypeError, ValueError):
                row.append(None)
        rows.append(row)
    return rows


def step_rank_logps(meta: dict, key: str) -> List[Optional[List[float]]]:
    """Length-20 log-probs aligned with ``step_c``.

    ``key`` is ``last_rank_logp``, ``penult_rank_logp``, or
    ``penult_on_last_logp``. Steps whose ``last_c`` entry is None are dropped.
    """
    last_c = list(meta.get("last_c") or [])
    column = list(meta.get(key) or [])
    out: List[Optional[List[float]]] = []
    for i, c in enumerate(last_c):
        if c is None:
            continue
        raw = column[i] if i < len(column) else None
        if not raw:
            out.append(None)
            continue
        try:
            out.append([float(x) for x in raw])
        except (TypeError, ValueError):
            out.append(None)
    return out


def step_sampled_tokens(meta: dict) -> List[Optional[int]]:
    """Sampled token id aligned with ``step_c`` / ``step_dc``.

    ``sampled_token`` is appended once per decode step that has ``last_c``.
    Steps whose ``last_c`` entry is None are dropped, same as ``step_values``.
    """
    last_c = list(meta.get("last_c") or [])
    sampled = list(meta.get("sampled_token") or [])
    out: List[Optional[int]] = []
    for i, c in enumerate(last_c):
        if c is None:
            continue
        raw = sampled[i] if i < len(sampled) else None
        if raw is None:
            out.append(None)
            continue
        try:
            out.append(int(raw))
        except (TypeError, ValueError):
            out.append(None)
    return out


def mean_overlap_columns(rows: Sequence[Sequence[Optional[float]]]) -> List[Optional[float]]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    out: List[Optional[float]] = []
    for j in range(width):
        vals = []
        for row in rows:
            if j < len(row) and row[j] is not None:
                vals.append(float(row[j]))
        out.append(sum(vals) / len(vals) if vals else None)
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
