"""Table 1 voting: windows, negative-dynamics shift, and confidence filtering."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from confdyn.table_vote import (  # noqa: E402
    Trace,
    filter_then_weight,
    score_traces,
    select_window,
    weighted_correct,
    window_starts,
)


def _trace(slot, key, correct, c=1.0, tc=1.0, d=1.0, td=1.0):
    return Trace(slot, key, correct, c, tc, d, td)


def test_paper_window_starts():
    assert window_starts() == [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
    ]


def test_window_uses_original_slots():
    traces = [_trace(0, "a", 1), _trace(10, "b", 0), _trace(30, "c", 1)]
    got = select_window(traces, 0, 30)
    assert [trace.slot for trace in got] == [0, 10]


def test_negative_dynamics_are_shifted_before_weighting():
    # Raw sums: A = 1, B = -3 + 2 = -1, so A wins.
    # After subtracting the minimum (-3): A = 4, B = 0 + 5, so B wins.
    keys = ["A", "B", "B"]
    corrects = [0.0, 1.0, 1.0]
    weights = [1.0, -3.0, 2.0]
    assert weighted_correct(keys, corrects, weights, shift_if_negative=False) == 0
    assert weighted_correct(keys, corrects, weights, shift_if_negative=True) == 1


def test_positive_dynamics_are_not_shifted():
    keys = ["A", "B", "B"]
    corrects = [0.0, 1.0, 1.0]
    # 3 > 1+1, so A wins. A min-shift would make B win (2+2 > 2).
    weights = [3.0, 1.0, 1.0]
    assert weighted_correct(keys, corrects, weights, shift_if_negative=True) == 0


def test_nonpositive_pool_falls_back_to_majority():
    keys = ["A", "B", "B"]
    corrects = [0.0, 1.0, 1.0]
    assert weighted_correct(keys, corrects, [-2.0, -1.0, 0.0], shift_if_negative=False) == 1


def test_tail_confidence_filter_then_dynamics_weight():
    # Top 50% by tail confidence keeps the two high-C traces.
    # Their dynamics weights then select the correct answer.
    traces = [
        _trace(0, "wrong", 0, tc=10, td=0.2),
        _trace(1, "right", 1, tc=9, td=3.0),
        _trace(2, "wrong", 0, tc=1, td=5.0),
        _trace(3, "wrong", 0, tc=0, td=4.0),
    ]
    scored = score_traces(traces)
    assert scored["ctf50"] == 0  # both retained answers are one vote; earliest key wins
    assert scored["ctf_dtw50"] == 1
    assert scored["standard"] == 0


def test_filter_then_weight_helper_keeps_highest_half():
    keys = ["a", "b", "b", "a"]
    corrects = [0.0, 1.0, 1.0, 0.0]
    scores = [1.0, 9.0, 8.0, 2.0]
    assert filter_then_weight(
        keys, corrects, scores, scores, 0.5, shift_if_negative=False, same_score=True
    ) == 1
