"""Post-hoc voting used for the main results table (Section 4.2, Table 1).

Each problem has a pool of up to N=50 finished trajectories. Accuracy is the
mean over 20 windows of 30 original sample slots, then the macro-average
across benchmarks. Length-capped trajectories are already absent from the
pool; windows are indexed by the original sample slot, so a window can contain
fewer than 30 traces.

Weights
-------
Confidence weights are used as stored. Dynamics weights are shifted by the
minimum inside the voting pool only when that pool contains a negative value,
which makes every weight non-negative while leaving an all-positive pool
unchanged. A pool whose weights are all non-positive falls back to unweighted
majority vote. Ties keep the earliest trajectory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Iterable, List, Optional, Sequence

import confdyn.filter_metrics as fm


POOL_SIZE = 50
WINDOW_WIDTH = 30
N_WINDOWS = 20
TAIL_TOKENS = 2048
FILTER_RATIOS = (Decimal("0.8"), Decimal("0.5"))

# Column order matches Table 1: direct voting, then top-80%, then top-50%.
COLUMN_KEYS: tuple[str, ...] = (
    "standard",
    "cw",
    "ctw",
    "dw",
    "dtw",
    "cf80",
    "ctf80",
    "cf_dtw80",
    "ctf_dtw80",
    "cf50",
    "ctf50",
    "cf_dtw50",
    "ctf_dtw50",
)

DIRECT_KEYS = ("standard", "cw", "ctw", "dw", "dtw")
TOP80_KEYS = ("cf80", "ctf80", "cf_dtw80", "ctf_dtw80")
TOP50_KEYS = ("cf50", "ctf50", "cf_dtw50", "ctf_dtw50")

COLUMN_LABELS = {
    "standard": "Standard",
    "cw": "CW",
    "ctw": "C^T W",
    "dw": "DW",
    "dtw": "D^T W",
    "cf80": "CF-CW",
    "ctf80": "C^T F-C^T W",
    "cf_dtw80": "CF-D^T W",
    "ctf_dtw80": "C^T F-D^T W",
    "cf50": "CF-CW",
    "ctf50": "C^T F-C^T W",
    "cf_dtw50": "CF-D^T W",
    "ctf_dtw50": "C^T F-D^T W",
}


@dataclass(frozen=True)
class Trace:
    """One finished trajectory inside a problem's sample pool."""

    slot: int
    key: object
    correct: float
    confidence: Optional[float]
    tail_confidence: Optional[float]
    dynamics: Optional[float]
    tail_dynamics: Optional[float]


def window_starts(
    pool: int = POOL_SIZE,
    width: int = WINDOW_WIDTH,
    n_windows: int = N_WINDOWS,
) -> List[int]:
    """Starts of ``n_windows`` slots of length ``width`` inside ``0 .. pool``.

    Starts are ``round_half_up(i * (pool - width) / (n_windows - 1))``.
    For the paper setting this is
    0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20.
    """
    if n_windows < 1:
        raise ValueError("n_windows must be positive")
    if n_windows == 1 or pool == width:
        return [0]
    span = Decimal(pool - width) / Decimal(n_windows - 1)
    return [
        int((Decimal(i) * span).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        for i in range(n_windows)
    ]


def select_window(traces: Sequence[Trace], start: int, width: int = WINDOW_WIDTH) -> List[Trace]:
    end = start + width
    return [trace for trace in traces if start <= trace.slot < end]


def _finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _passed(correct: float) -> int:
    return 1 if float(correct) > 0 else 0


def _weights(values: Sequence[Optional[float]]) -> List[float]:
    out = []
    for value in values:
        number = _finite(value)
        out.append(0.0 if number is None else number)
    return out


def _majority_correct(keys: Sequence, corrects: Sequence[float]) -> int:
    if not keys:
        return 0
    _, index, _ = fm.majority_vote(keys)
    return _passed(corrects[index])


def weighted_correct(
    keys: Sequence,
    corrects: Sequence[float],
    weights: Sequence[Optional[float]],
    *,
    shift_if_negative: bool,
) -> int:
    """Weighted majority. Dynamics pools are shifted only if some weight is negative."""
    if not keys:
        return 0
    numeric = _weights(weights)
    if shift_if_negative and any(weight < 0 for weight in numeric):
        floor = min(numeric)
        numeric = [weight - floor for weight in numeric]
    if all(weight <= 0 for weight in numeric):
        return _majority_correct(keys, corrects)
    _, index = fm.weighted_majority_vote(keys, numeric)
    return _passed(corrects[index])


def _filter_indices(scores: Sequence[Optional[float]], rho: float) -> List[int]:
    n = len(scores)
    if n == 0:
        return []
    k = max(1, int(math.ceil(n * float(rho))))
    return fm.topk_indices([_finite(score) for score in scores], k, highest=True)


def filter_then_weight(
    keys: Sequence,
    corrects: Sequence[float],
    filter_scores: Sequence[Optional[float]],
    weights: Sequence[Optional[float]],
    rho: float,
    *,
    shift_if_negative: bool,
    same_score: bool = False,
) -> int:
    """Keep the top ``rho`` by ``filter_scores``, then weighted-vote that subset."""
    n = len(keys)
    if n == 0:
        return 0
    indices = _filter_indices(filter_scores, rho)
    if not indices:
        if same_score:
            indices = list(range(n))
        else:
            return _majority_correct(keys, corrects)
    return weighted_correct(
        [keys[i] for i in indices],
        [corrects[i] for i in indices],
        [weights[i] for i in indices],
        shift_if_negative=shift_if_negative,
    )


def score_traces(traces: Sequence[Trace]) -> Dict[str, int]:
    """One problem, one window. Each value is 1 when the voted answer is correct."""
    keys = [trace.key for trace in traces]
    corrects = [trace.correct for trace in traces]
    confidence = [trace.confidence for trace in traces]
    tail_c = [trace.tail_confidence for trace in traces]
    dynamics = [trace.dynamics for trace in traces]
    tail_d = [trace.tail_dynamics for trace in traces]
    out = {
        "standard": _majority_correct(keys, corrects),
        "cw": weighted_correct(keys, corrects, confidence, shift_if_negative=False),
        "ctw": weighted_correct(keys, corrects, tail_c, shift_if_negative=False),
        "dw": weighted_correct(keys, corrects, dynamics, shift_if_negative=True),
        "dtw": weighted_correct(keys, corrects, tail_d, shift_if_negative=True),
    }
    for rho, tag in ((0.8, "80"), (0.5, "50")):
        out[f"cf{tag}"] = filter_then_weight(
            keys, corrects, confidence, confidence, rho,
            shift_if_negative=False, same_score=True,
        )
        out[f"ctf{tag}"] = filter_then_weight(
            keys, corrects, tail_c, tail_c, rho,
            shift_if_negative=False, same_score=True,
        )
        out[f"cf_dtw{tag}"] = filter_then_weight(
            keys, corrects, confidence, tail_d, rho, shift_if_negative=True,
        )
        out[f"ctf_dtw{tag}"] = filter_then_weight(
            keys, corrects, tail_c, tail_d, rho, shift_if_negative=True,
        )
    return out


def mean_accuracy(problems: Sequence[Sequence[Trace]], starts: Sequence[int]) -> Dict[str, Decimal]:
    """Mean accuracy in percent over problems and windows. Not yet rounded."""
    totals = {key: Decimal(0) for key in COLUMN_KEYS}
    n_problems = Decimal(len(problems))
    n_windows = Decimal(len(starts))
    if n_problems == 0 or n_windows == 0:
        return totals
    for start in starts:
        window_sum = {key: Decimal(0) for key in COLUMN_KEYS}
        for traces in problems:
            scored = score_traces(select_window(traces, start))
            for key in COLUMN_KEYS:
                window_sum[key] += Decimal(scored[key])
        for key in COLUMN_KEYS:
            totals[key] += window_sum[key] / n_problems * Decimal(100)
    return {key: totals[key] / n_windows for key in COLUMN_KEYS}


def round_half_up(value: Decimal, places: str = "0.01") -> Decimal:
    return value.quantize(Decimal(places), rounding=ROUND_HALF_UP)


def macro_average(rows: Iterable[Dict[str, Decimal]]) -> Dict[str, Decimal]:
    rows = list(rows)
    if not rows:
        return {key: Decimal(0) for key in COLUMN_KEYS}
    n = Decimal(len(rows))
    return {
        key: sum((row[key] for row in rows), Decimal(0)) / n
        for key in COLUMN_KEYS
    }
