"""CPU tests for majority vote / over-generate-then-filter helpers."""

import json
import math
import os
import pathlib
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from confdyn.filter_metrics import (  # noqa: E402
    avg_or_none,
    avg_select_k,
    confidence_weights,
    majority_vote,
    mean_steps,
    mean_steps_where,
    mean_tail_frac,
    mean_tail_steps,
    split_by_score,
    step_values,
    step_layer_confidences,
    step_layer_top_overlap,
    step_layer_top_vectors,
    step_sampled_tokens,
    subset_pass1,
    vote_on_subset,
    weighted_majority_vote,
    weighted_vote_acc,
)


def test_majority_vote_picks_first_of_ties():
    voted, idx, counts = majority_vote(["7", "8", "7", "9"])
    assert voted == "7"
    assert idx == 0
    assert counts["7"] == 2


def test_majority_vote_uses_mode():
    voted, idx, _ = majority_vote(["a", "b", "b", "c"])
    assert voted == "b"
    assert idx == 1


def test_weighted_vote_prefers_confident_minority():
    keys = ["wrong", "wrong", "right"]
    weights = [0.1, 0.1, 5.0]
    voted, idx = weighted_majority_vote(keys, weights)
    assert voted == "right"
    assert idx == 2


def test_split_by_score_high_and_low():
    pass_list = [1.0, 0.0, 1.0, 0.0]
    scores = [0.1, 0.9, 0.2, 0.8]
    p1_hi, pk_hi, p1_rest, pk_rest = split_by_score(pass_list, scores, k=2, highest=True)
    assert math.isclose(p1_hi, 0.0)
    assert math.isclose(pk_hi, 0.0)
    assert math.isclose(p1_rest, 1.0)
    p1_lo, _, _, _ = split_by_score(pass_list, scores, k=2, highest=False)
    assert math.isclose(p1_lo, 1.0)


def test_filter_then_majority_vote():
    pass_list = [0.0, 1.0, 1.0, 0.0]
    keys = ["x", "y", "y", "x"]
    ents = [3.0, 0.2, 0.3, 2.5]
    lo = sorted(range(4), key=lambda i: ents[i])[:2]
    acc = vote_on_subset(pass_list, keys, lo)
    assert acc == 1.0


def test_confidence_weights_from_logprob():
    w = confidence_weights([-0.1, -2.0], [1.0, 1.0])
    assert w[0] > w[1]


def test_weighted_vote_acc():
    acc = weighted_vote_acc([0.0, 1.0], ["a", "b"], [0.1, 10.0])
    assert acc == 1.0


def test_untruncated_indices():
    from confdyn.filter_metrics import untruncated_indices

    # length cap, explicit non-stop, and unknown finish under the cap
    idx = untruncated_indices(
        [True, False, None, True],
        [100, 50, 40000, 10],
        40000,
    )
    assert idx == [0, 3]


def test_subset_pass1():
    assert math.isclose(subset_pass1([1.0, 0.0, 1.0], [0.2, 0.9, 0.1], 2, False), 1.0)


def test_mean_steps_and_match():
    assert math.isclose(mean_steps([0.2, None, 0.4]), 0.3)
    assert mean_steps([]) is None
    assert math.isclose(mean_steps_where([1.0, 2.0, 3.0], [1, 0, 1]), 2.0)
    assert math.isclose(mean_steps_where([1.0, 2.0, 3.0], [1, 0, 1], invert=True), 2.0)
    assert math.isclose(mean_tail_steps([1.0, 2.0, 3.0, 4.0], 2), 3.5)
    assert math.isclose(mean_tail_steps([1.0, None, 2.0], 2048), 1.5)
    assert mean_tail_steps([]) is None
    assert math.isclose(mean_tail_frac([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.20), 9.5)
    assert math.isclose(mean_tail_frac(list(range(1, 11)), 0.10), 10.0)


def test_switch_select_k():
    from confdyn.filter_metrics import avg_select_k, step_match_flags

    assert step_match_flags([1.0, 0.0, None, 0.9]) == [1, 0, 1]
    rows = [
        {
            "passat1_list": [1.0, 0.0, 1.0, 0.0],
            "mean_switch_dmp_list": [0.1, 0.9, 0.2, 0.8],
        }
    ]
    # picks 0.9 and 0.8 → both wrong
    assert math.isclose(avg_select_k(rows, "mean_switch_dmp_list", 2), 0.0)


def test_meta_means_switch():
    import confdyn.generate as gen

    meta = {
        "last_mp": [1.0, 1.0],
        "last_mcos": [0.5, 0.5],
        "last_c": [0.1, 0.1],
        "penult_dmp": [1.0, 3.0, 2.0],
        "penult_dmc": [0.2, 0.8, 0.4],
        "penult_dc": [0.0, 1.0, 0.0],
        "penult_match": [1, 0, 1],
    }
    m = gen._meta_means(meta, tail_window=2)
    assert math.isclose(m["mean_all_dmp"], 2.0)
    assert math.isclose(m["mean_match_dmp"], 1.5)
    assert math.isclose(m["mean_switch_dmp"], 3.0)
    assert math.isclose(m["mean_switch_dmc"], 0.8)
    assert math.isclose(m["mean_tail_dmp"], 2.5)
    assert math.isclose(m["mean_tail_dmc"], 0.6)
    assert math.isclose(m["mean_tail_c"], 0.1)
    # 10% of 3 steps → 1 step; 20% → 1 step (ceil)
    assert math.isclose(m["mean_tail10_dmp"], 2.0)
    assert math.isclose(m["mean_tail20_dmp"], 2.0)


def test_sheet_columns_select_highest():
    pass_list = [1.0, 0.0, 1.0, 0.0]
    last_mp = [0.1, 0.9, 0.2, 0.8]
    last_c = [0.4, 0.1, 0.5, 0.2]
    p1_mp, _, _, _ = split_by_score(pass_list, last_mp, k=2, highest=True)
    p1_c, _, _, _ = split_by_score(pass_list, last_c, k=2, highest=True)
    assert math.isclose(p1_mp, 0.0)
    assert math.isclose(p1_c, 1.0)
    rows = [{"passat1_hi_last_mp": 0.5}, {"passat1_hi_last_mp": 1.0}]
    assert math.isclose(avg_or_none(rows, "passat1_hi_last_mp"), 0.75)


def test_step_values_drops_none_and_rounds():
    got = step_values([None, 0.123456789, None, -0.5])
    assert got == [0.123457, -0.5]


def test_step_layer_confidences_aligns_to_last_layer():
    meta = {
        "last_c": [None, 1.0, 1.2],
        "layer_c_1": [None, 0.5, 0.7],
        "layer_c_2": [None, 0.25, None],
    }
    assert step_layer_confidences(meta, depth=3) == [
        [0.25, 0.5, 1.0],
        [None, 0.7, 1.2],
    ]


def test_step_layer_top_overlap_with_last_layer():
    meta = {
        "last_c": [None, 1.0, 1.2],
        "layer_top_0": [None, [1, 2, 3], [4, 5, 6]],
        "layer_top_1": [None, [1, 2, 9], [4, 5, 0]],
        "layer_top_2": [None, [9, 8, 7], [4, 0, 1]],
    }
    tops, overlaps = step_layer_top_overlap(meta, depth=3)
    assert tops == [
        [[9, 8, 7], [1, 2, 9], [1, 2, 3]],
        [[4, 0, 1], [4, 5, 0], [4, 5, 6]],
    ]
    assert overlaps[0] == [0.0, round(2 / 3, 6), 1.0]
    assert overlaps[1][0] == round(1 / 3, 6)
    assert overlaps[1][1] == round(2 / 3, 6)
    assert overlaps[1][2] == 1.0


def test_step_layer_top_vectors_follow_last_c():
    meta = {
        "last_c": [None, 1.0],
        "layer_top_logit_0": [[9.0], [1.5]],
        "layer_top_logit_1": [[8.0], [0.5]],
    }
    assert step_layer_top_vectors(meta, "layer_top_logit_", depth=2) == [
        [[0.5], [1.5]]
    ]


def test_step_sampled_tokens_follow_last_c():
    meta = {
        "last_c": [None, 1.0, 1.2],
        "sampled_token": [9, 22, 13],
    }
    assert step_sampled_tokens(meta) == [22, 13]


def test_step_sidecar_pops_lists(tmp_path):
    import confdyn.generate as gen

    payload = {
        "results": [
            {
                "idx": 0,
                "sample_deltas": [
                    {
                        "sample": 0,
                        "step_c": [1.0, 1.1, 1.2],
                        "step_dc": [0.01, 0.02],
                        "step_dc_n3": [0.03],
                        "step_dc_n4": [0.04, 0.05],
                        "step_match": [1, 0],
                        "step_layer_c": [[0.1, 1.0], [0.2, 1.1], [0.3, 1.2]],
                    }
                ],
            }
        ]
    }
    dump = str(tmp_path / "out.json")
    path = gen._write_step_sidecar(dump, payload)
    assert path.endswith(".steps.jsonl")
    rec = json.loads(pathlib.Path(path).read_text().splitlines()[0])
    assert rec["idx"] == 0
    assert rec["sample"] == 0
    assert rec["steps"] == 3
    assert rec["step_c"] == [1.0, 1.1, 1.2]
    assert rec["step_dc"] == [0.01, 0.02]
    assert rec["step_dc_n3"] == [0.03]
    assert rec["step_dc_n4"] == [0.04, 0.05]
    assert rec["step_match"] == [1, 0]
    assert rec["step_layer_c"] == [[0.1, 1.0], [0.2, 1.1], [0.3, 1.2]]
    assert "step_c" not in payload["results"][0]["sample_deltas"][0]
    assert "step_layer_c" not in payload["results"][0]["sample_deltas"][0]
    assert "step_match" not in payload["results"][0]["sample_deltas"][0]
