"""CPU checks for ΔC chunk tree search: 3→2, then 6→2, fixed chunk length."""

import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from confdyn.engine import _cuda_graph_max_bs
from confdyn.tree_search import (
    expansion_seed,
    parse_expansion_seed,
    run_dc_chunk_search,
    select_keep_c,
    select_keep_c2dc,
    select_keep_c4dc,
    select_keep_cguard,
    select_random,
    select_top,
    selection_seed,
)


class _Node:
    def __init__(self, dc):
        self.chunk_dc = dc


class _Tok:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [7, 8, 9]

    def decode(self, ids, skip_special_tokens=True):
        return ",".join(str(i) for i in ids)


def _args(**kwargs):
    base = dict(
        tree_beam=2,
        tree_branch=3,
        tree_chunk=500,
        max_new_tokens=1000,
        max_batch=64,
        random_seed=0,
        start_idx=0,
        return_logprob=True,
        top_logprobs_num=0,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        dataset="aime",
        end_idx=20,
        num_samples=1,
        cuda_graph_max_bs=None,
        dp_size=1,
        tree_search=True,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _script(table, chunk_stop_len=4, branch_factor=3):
    calls = []
    batches = []

    def generate(input_ids, sps):
        batches.append(len(input_ids))
        outs = []
        for ids, sp in zip(input_ids, sps):
            case, rnd, slot, branch = parse_expansion_seed(
                sp["sampling_seed"], 0, branch_factor
            )
            spec = table[(case, rnd, slot, branch)]
            stop = bool(spec.get("stop"))
            n_out = chunk_stop_len if stop else int(sp["max_new_tokens"])
            dc = spec.get("dc")
            calls.append(
                {
                    "case": case,
                    "round": rnd,
                    "slot": slot,
                    "branch": branch,
                    "n_new": int(sp["max_new_tokens"]),
                    "in_len": len(ids),
                    "stop": stop,
                }
            )
            step_dc = [None] * n_out if dc is None else [float(dc)] * n_out
            c = spec.get("c", 0.1)
            step_c = [None] * n_out if c is None else [float(c)] * n_out
            outs.append(
                {
                    "output_ids": [10_000 * (rnd + 1) + 100 * slot + branch] * n_out,
                    "meta_info": {
                        "penult_dc": step_dc,
                        "last_c": step_c,
                        "finish_reason": {"type": "stop" if stop else "length"},
                        "completion_tokens": n_out,
                    },
                }
            )
        return outs

    return generate, calls, batches


def test_select_top_is_global_and_stable():
    kids = [_Node(0.2), _Node(0.9), _Node(None), _Node(0.9)]
    picked = select_top(kids, 2)
    assert [k.chunk_dc for k in picked] == [0.9, 0.9]
    assert picked[0] is kids[1]
    assert picked[1] is kids[3]


def test_seed_roundtrip():
    seed = expansion_seed(8, case_idx=41, round_idx=7, parent_slot=1, branch=2, branch_factor=3)
    assert parse_expansion_seed(seed, 8, 3) == (41, 7, 1, 2)


def test_first_round_three_then_global_top_two_of_six():
    # Round 0 keeps branches 1 (0.9) and 2 (0.4), in that order.
    # Round 1 is global: both survivors come from slot 1, not one per parent.
    table = {
        (0, 0, 0, 0): {"dc": 0.1},
        (0, 0, 0, 1): {"dc": 0.9},
        (0, 0, 0, 2): {"dc": 0.4},
        (0, 1, 0, 0): {"dc": 0.2},
        (0, 1, 0, 1): {"dc": 0.3},
        (0, 1, 0, 2): {"dc": 0.15},
        (0, 1, 1, 0): {"dc": 0.95},
        (0, 1, 1, 1): {"dc": 0.05},
        (0, 1, 1, 2): {"dc": 0.7},
    }
    generate, calls, batches = _script(table)
    result = run_dc_chunk_search(
        None, _Tok(), ["prompt"], _args(max_new_tokens=1000, tree_chunk=500), "qwen3", generate
    )
    assert [c["n_new"] for c in calls[:3]] == [500, 500, 500]
    assert len([c for c in calls if c["round"] == 0]) == 3
    assert len([c for c in calls if c["round"] == 1]) == 6
    assert all(c["n_new"] == 500 for c in calls if c["round"] == 1)
    assert [c["in_len"] for c in calls if c["round"] == 1] == [503] * 6
    finals = result.bundles[0]
    # Round 0 drops branch 0. Round 1 is the last round, so all 6 stay.
    assert len(finals) == 6
    assert [t["meta"]["tree_chunk_dc"] for t in finals[:2]] == [
        [0.4, 0.95],
        [0.4, 0.7],
    ]
    assert [t["meta"]["tree_branch_path"] for t in finals] == [
        [2, 0],
        [2, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 1],
    ]
    assert [t["meta"]["tree_selected"] for t in finals] == [True, True, False, False, False, False]
    assert all(t["completion_tokens"] == 1000 for t in finals)
    assert all(len(t["meta"]["penult_dc"]) == 1000 for t in finals)
    assert result.case_info[0]["n_pruned"] == 1
    assert result.case_info[0]["round_log"][1]["last_round"] is True
    assert result.case_info[0]["round_log"][1]["n_extra"] == 4
    assert result.rounds == 2


def test_selected_eos_occupies_a_slot_and_is_not_backfilled():
    table = {
        (0, 0, 0, 0): {"dc": 0.9, "stop": True},
        (0, 0, 0, 1): {"dc": 0.8, "stop": True},
        (0, 0, 0, 2): {"dc": 0.7},
    }
    generate, calls, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=2000, tree_chunk=500),
        "qwen3",
        generate,
    )
    assert len(calls) == 3
    # Both selected chunks already finished, so this is the last round.
    # The unselected open chunk is kept and is not extended in their place.
    assert [t["meta"]["tree_chunk_dc"] for t in result.bundles[0]] == [[0.9], [0.8], [0.7]]
    assert [t["meta"]["tree_selected"] for t in result.bundles[0]] == [True, True, False]
    assert [t["finished"] for t in result.bundles[0]] == [True, True, False]
    assert [t["completion_tokens"] for t in result.bundles[0]] == [4, 4, 500]
    assert result.case_info[0]["n_pruned"] == 0


def test_eos_survivor_is_kept_and_only_the_open_beam_expands():
    table = {
        (0, 0, 0, 0): {"dc": 0.1},
        (0, 0, 0, 1): {"dc": 0.9},
        (0, 0, 0, 2): {"dc": 0.4},
        (0, 1, 0, 0): {"dc": 0.2},
        (0, 1, 0, 1): {"dc": 0.3},
        (0, 1, 0, 2): {"dc": 0.15},
        (0, 1, 1, 0): {"dc": 0.95, "stop": True},
        (0, 1, 1, 1): {"dc": 0.05},
        (0, 1, 1, 2): {"dc": 0.7},
        (0, 2, 0, 0): {"dc": 0.4},
        (0, 2, 0, 1): {"dc": 0.8},
        (0, 2, 0, 2): {"dc": 0.1},
    }
    generate, calls, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=1500, tree_chunk=500),
        "qwen3",
        generate,
    )
    by_round = {}
    for call in calls:
        by_round.setdefault(call["round"], []).append(call)
    assert len(by_round[0]) == 3
    assert len(by_round[1]) == 6
    assert len(by_round[2]) == 3
    assert [c["in_len"] for c in by_round[2]] == [1003, 1003, 1003]
    assert [c["n_new"] for c in by_round[2]] == [500, 500, 500]
    scores = [t["meta"]["tree_chunk_dc"] for t in result.bundles[0]]
    assert scores[0] == [0.4, 0.95]
    assert result.bundles[0][0]["finished"] is True
    assert result.bundles[0][0]["meta"]["tree_selected"] is True
    assert result.bundles[0][0]["completion_tokens"] == 504
    assert scores[1:] == [[0.4, 0.7, 0.8], [0.4, 0.7, 0.4], [0.4, 0.7, 0.1]]
    assert [t["meta"]["tree_selected"] for t in result.bundles[0][1:]] == [True, True, False]
    assert all(t["completion_tokens"] == 1500 for t in result.bundles[0][1:])


def test_low_dc_eos_is_dropped_before_the_last_round():
    table = {
        (0, 0, 0, 0): {"dc": 0.01, "stop": True},
        (0, 0, 0, 1): {"dc": 0.5},
        (0, 0, 0, 2): {"dc": 0.4},
    }
    for slot in (0, 1):
        for branch in range(3):
            table[(0, 1, slot, branch)] = {"dc": 0.2 + 0.1 * slot + 0.01 * branch}
    generate, calls, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=1000, tree_chunk=500),
        "qwen3",
        generate,
    )
    assert len([c for c in calls if c["round"] == 0]) == 3
    assert len([c for c in calls if c["round"] == 1]) == 6
    finals = result.bundles[0]
    assert len(finals) == 6
    assert all(t["meta"]["tree_chunk_dc"][0] != 0.01 for t in finals)
    assert all(t["meta"]["tree_branch_path"][0] in (1, 2) for t in finals)
    assert result.case_info[0]["n_pruned"] == 1


def test_cases_are_selected_independently_and_batches_split():
    table = {}
    for branch, dc in enumerate((0.9, 0.1, 0.2)):
        table[(0, 0, 0, branch)] = {"dc": dc}
    for branch, dc in enumerate((0.05, 0.06, 0.99)):
        table[(1, 0, 0, branch)] = {"dc": dc}
    generate, calls, batches = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["a", "b"],
        _args(max_new_tokens=500, tree_chunk=500, max_batch=4),
        "qwen3",
        generate,
    )
    assert batches == [4, 2]
    assert len(calls) == 6
    # One round is the last round, so the unselected chunk is kept after the beam.
    assert [t["meta"]["tree_chunk_dc"] for t in result.bundles[0]] == [[0.9], [0.2], [0.1]]
    assert [t["meta"]["tree_selected"] for t in result.bundles[0]] == [True, True, False]
    assert [t["meta"]["tree_chunk_dc"] for t in result.bundles[1]] == [[0.99], [0.06], [0.05]]
    assert result.case_info[0]["round_log"][0]["selected_dc"] == [0.9, 0.2]
    assert result.case_info[1]["round_log"][0]["selected_dc"] == [0.99, 0.06]


def test_last_chunk_uses_the_remaining_budget():
    table = {
        (0, 0, 0, 0): {"dc": 0.2},
        (0, 0, 0, 1): {"dc": 0.8},
        (0, 0, 0, 2): {"dc": 0.4},
    }
    for slot in (0, 1):
        for branch, dc in enumerate((0.1, 0.2, 0.3)):
            table[(0, 1, slot, branch)] = {"dc": dc + slot}
    generate, calls, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=600, tree_chunk=500),
        "qwen3",
        generate,
    )
    assert [c["n_new"] for c in calls if c["round"] == 0] == [500, 500, 500]
    assert [c["n_new"] for c in calls if c["round"] == 1] == [100] * 6
    assert len(result.bundles[0]) == 6
    assert all(t["completion_tokens"] == 600 for t in result.bundles[0])
    assert all(len(t["meta"]["penult_dc"]) == 600 for t in result.bundles[0])
    assert result.case_info[0]["n_pruned"] == 1


def test_missing_dc_is_not_extended():
    table = {
        (0, 0, 0, 0): {"dc": None},
        (0, 0, 0, 1): {"dc": 0.2},
        (0, 0, 0, 2): {"dc": 0.1},
    }
    for slot in (0, 1):
        for branch in range(3):
            table[(0, 1, slot, branch)] = {"dc": 0.3}
    generate, calls, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=1000, tree_chunk=500),
        "qwen3",
        generate,
    )
    assert len([c for c in calls if c["round"] == 1]) == 6
    finals = result.bundles[0]
    assert len(finals) == 6
    assert all(t["meta"]["tree_branch_path"][0] != 0 for t in finals)


def test_final_layer_c_outranks_delta_c():
    # ΔC would keep branches 0 and 2. Final-layer C keeps 1 and 2.
    table = {
        (0, 0, 0, 0): {"dc": 0.9, "c": 0.1},
        (0, 0, 0, 1): {"dc": 0.2, "c": 0.8},
        (0, 0, 0, 2): {"dc": 0.4, "c": 0.5},
    }
    generate, calls, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=500, tree_chunk=500, tree_score="c"),
        "qwen3",
        generate,
    )
    assert len(calls) == 3
    finals = result.bundles[0]
    assert [t["meta"]["tree_score"] for t in finals] == ["c", "c", "c"]
    assert [t["meta"]["tree_branch_path"] for t in finals] == [[1], [2], [0]]
    assert [t["meta"]["tree_selected"] for t in finals] == [True, True, False]
    assert [t["meta"]["tree_chunk_dc"] for t in finals] == [[0.8], [0.5], [0.1]]
    assert result.case_info[0]["score"] == "c"
    assert result.case_info[0]["round_log"][0]["selected_dc"] == [0.8, 0.5]


def test_random_keeps_two_and_ignores_the_score():
    import random

    table = {
        (0, 0, 0, 0): {"dc": 0.9, "c": 0.1},
        (0, 0, 0, 1): {"dc": 0.2, "c": 0.8},
        (0, 0, 0, 2): {"dc": 0.1, "c": 0.05},
    }
    expected = random.Random(selection_seed(0, 0, 0)).sample(range(3), 2)
    leftover = [i for i in range(3) if i not in expected]
    generate, calls, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=500, tree_chunk=500, tree_score="random"),
        "qwen3",
        generate,
    )
    assert expected != [0, 1]
    assert len(calls) == 3
    finals = result.bundles[0]
    paths = [t["meta"]["tree_branch_path"] for t in finals]
    assert paths == [[i] for i in expected] + [[i] for i in leftover]
    assert [t["meta"]["tree_selected"] for t in finals] == [True, True, False]
    assert [t["meta"]["tree_score"] for t in finals] == ["random", "random", "random"]
    kept = list(expected) + leftover
    # Recorded score is still chunk-mean ΔC, not the draw.
    assert [t["meta"]["tree_chunk_dc"] for t in finals] == [
        [table[(0, 0, 0, i)]["dc"]] for i in kept
    ]
    assert result.case_info[0]["score"] == "random"
    kids = [_Node(0.9), _Node(0.1), _Node(0.2)]
    again = select_random(kids, 2, random.Random(selection_seed(0, 0, 0)))
    assert [kids.index(node) for node in again] == expected


def test_warmup_is_the_first_round_only():
    table = {}
    for branch, dc in enumerate((0.2, 0.9, 0.4)):
        table[(0, 0, 0, branch)] = {"dc": dc}
    for slot in (0, 1):
        for branch in range(3):
            table[(0, 1, slot, branch)] = {"dc": 0.3}
    generate, calls, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=3500, tree_chunk=500, tree_warmup=3000, tree_rank="tail", tree_tail=2048),
        "qwen3",
        generate,
    )
    assert [c["n_new"] for c in calls if c["round"] == 0] == [3000, 3000, 3000]
    assert [c["n_new"] for c in calls if c["round"] == 1] == [500] * 6
    assert result.case_info[0]["warmup"] == 3000
    assert result.case_info[0]["rank"] == "tail"
    assert all(t["meta"]["tree_rank"] == "tail" for t in result.bundles[0])
    assert len(result.bundles[0]) == 6


def test_tail_rank_uses_history_not_the_new_chunk():
    # First round: 4 tokens. Tail window 6, so the first score is the whole prefix.
    # Second round adds 2 tokens and hits the cap.
    # Chunk mean would keep the parent whose new chunk is 1.
    # Tail mean keeps the parent whose earlier tokens are 1, even though the new chunk is 0.
    table = {
        (0, 0, 0, 0): {"dc": 1.0},
        (0, 0, 0, 1): {"dc": 0.0},
        (0, 0, 0, 2): {"dc": 0.2},
    }
    for branch in range(3):
        table[(0, 1, 0, branch)] = {"dc": 0.0}
        table[(0, 1, 1, branch)] = {"dc": 1.0}
    generate, _, _ = _script(table)
    common = dict(max_new_tokens=6, tree_chunk=2, tree_warmup=4, tree_tail=6)
    tail = run_dc_chunk_search(
        None, _Tok(), ["prompt"], _args(tree_rank="tail", **common), "qwen3", generate
    )
    chunk = run_dc_chunk_search(
        None, _Tok(), ["prompt"], _args(tree_rank="chunk", **common), "qwen3", _script(table)[0]
    )
    tail_selected = [
        t["meta"]["tree_branch_path"]
        for t in tail.bundles[0]
        if t["meta"]["tree_selected"]
    ]
    chunk_selected = [
        t["meta"]["tree_branch_path"]
        for t in chunk.bundles[0]
        if t["meta"]["tree_selected"]
    ]
    assert tail_selected == [[0, 0], [0, 1]]
    assert chunk_selected == [[2, 0], [2, 1]]


def test_first_round_four_then_two_by_two():
    # Chunk-mean ΔC. Round 0 samples 4 and keeps branches 1 (0.9) and 3 (0.7).
    # Round 1 expands each into 2. That round hits the cap, so all 4 stay.
    table = {
        (0, 0, 0, 0): {"dc": 0.1},
        (0, 0, 0, 1): {"dc": 0.9},
        (0, 0, 0, 2): {"dc": 0.2},
        (0, 0, 0, 3): {"dc": 0.7},
    }
    for slot, base in ((0, 0.3), (1, 0.31)):
        table[(0, 1, slot, 0)] = {"dc": base}
        table[(0, 1, slot, 1)] = {"dc": base + 0.3}
    generate, calls, _ = _script(table, branch_factor=4)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(
            max_new_tokens=1000,
            tree_chunk=500,
            tree_beam=2,
            tree_branch=2,
            tree_init_branch=4,
        ),
        "qwen3",
        generate,
    )
    assert [c["n_new"] for c in calls if c["round"] == 0] == [500, 500, 500, 500]
    assert [c["n_new"] for c in calls if c["round"] == 1] == [500, 500, 500, 500]
    assert [c["in_len"] for c in calls if c["round"] == 1] == [503, 503, 503, 503]
    finals = result.bundles[0]
    assert [t["meta"]["tree_branch_path"] for t in finals] == [
        [3, 1],
        [1, 1],
        [1, 0],
        [3, 0],
    ]
    assert [t["meta"]["tree_selected"] for t in finals] == [True, True, False, False]
    assert result.case_info[0]["n_pruned"] == 2
    assert result.case_info[0]["init_branch"] == 4
    assert result.case_info[0]["branch"] == 2


def test_beam_one_keeps_one_of_three_each_round():
    # Every round samples 3 and continues only the highest chunk-mean ΔC.
    table = {
        (0, 0, 0, 0): {"dc": 0.2},
        (0, 0, 0, 1): {"dc": 0.8},
        (0, 0, 0, 2): {"dc": 0.4},
        (0, 1, 0, 0): {"dc": 0.1},
        (0, 1, 0, 1): {"dc": 0.3},
        (0, 1, 0, 2): {"dc": 0.9},
    }
    generate, calls, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=1000, tree_chunk=500, tree_beam=1, tree_branch=3),
        "qwen3",
        generate,
    )
    assert len([c for c in calls if c["round"] == 0]) == 3
    assert len([c for c in calls if c["round"] == 1]) == 3
    assert [c["in_len"] for c in calls if c["round"] == 1] == [503, 503, 503]
    finals = result.bundles[0]
    assert [t["meta"]["tree_branch_path"] for t in finals] == [[1, 2], [1, 0], [1, 1]]
    assert [t["meta"]["tree_selected"] for t in finals] == [True, False, False]
    assert result.case_info[0]["n_pruned"] == 2
    assert result.case_info[0]["beam"] == 1


class _Pair:
    def __init__(self, c, dc):
        self.chunk_c = c
        self.chunk_dc = dc


def test_keep_rules_pick_two_of_six():
    # C order is 0..5. Index 4 has the highest ΔC but sits at C rank 5.
    kids = [
        _Pair(0.9, 0.1),
        _Pair(0.8, 0.2),
        _Pair(0.7, 0.95),
        _Pair(0.6, 0.3),
        _Pair(0.5, 0.99),
        _Pair(0.1, 0.0),
    ]
    assert select_keep_c(kids, 2) == [kids[0], kids[1]]
    # Top 4 by C, then the two highest ΔC inside that pool. Index 4 is out.
    assert select_keep_c4dc(kids, 2) == [kids[2], kids[3]]
    # C rank 1 stays. The other slot is the best ΔC among C ranks 2–4.
    assert select_keep_cguard(kids, 2) == [kids[0], kids[2]]
    # One open slot: two-stage keeps the best ΔC in the C pool; the
    # conservative rule keeps C rank 1 only.
    assert select_keep_c4dc(kids, 1) == [kids[2]]
    assert select_keep_cguard(kids, 1) == [kids[0]]
    assert select_keep_c([_Pair(None, 0.9), _Pair(0.2, 0.0)], 1)[0].chunk_c == 0.2


def test_keep_c2dc_ranks_by_c_plus_two_dc():
    # Scores: 11, 10.2, 10, 8.8, 8. A missing score loses.
    kids = [
        _Pair(10.0, 0.0),
        _Pair(9.0, 0.6),
        _Pair(8.0, 0.4),
        _Pair(12.0, -2.0),
        _Pair(None, 100.0),
        _Pair(7.0, 2.0),
    ]
    assert select_keep_c2dc(kids, 2) == [kids[5], kids[1]]
    # A 2-point C lead beats a 0.5 ΔC lead: 10+0 > 8+2*0.5.
    close = [_Pair(10.0, 0.0), _Pair(8.0, 0.5)]
    assert select_keep_c2dc(close, 1) == [close[0]]


def test_keep_c1dc_ranks_by_c_plus_dc():
    # 10+0 beats 8+1.5. The same pair flips at alpha 2, where 8+3 = 11.
    kids = [_Pair(10.0, 0.0), _Pair(8.0, 1.5), _Pair(7.0, 0.2)]
    assert select_keep_c2dc(kids, 2, alpha=1.0) == [kids[0], kids[1]]
    assert select_keep_c2dc(kids, 2) == [kids[1], kids[0]]
    table = {
        (0, 0, 0, 0): {"dc": 0.0, "c": 10.0},
        (0, 0, 0, 1): {"dc": 1.5, "c": 8.0},
        (0, 0, 0, 2): {"dc": 0.2, "c": 7.0},
    }
    generate, _, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=500, tree_chunk=500, tree_keep="c1dc"),
        "qwen3",
        generate,
    )
    assert [t["meta"]["tree_branch_path"] for t in result.bundles[0]] == [[0], [1]]


def test_keep_c_drops_the_last_round_extra():
    table = {
        (0, 0, 0, 0): {"dc": 0.9, "c": 0.2},
        (0, 0, 0, 1): {"dc": 0.1, "c": 0.9},
        (0, 0, 0, 2): {"dc": 0.4, "c": 0.5},
    }
    generate, _, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=500, tree_chunk=500, tree_keep="c"),
        "qwen3",
        generate,
    )
    finals = result.bundles[0]
    assert len(finals) == 2
    assert [t["meta"]["tree_branch_path"] for t in finals] == [[1], [2]]
    assert [t["meta"]["tree_chunk_c"] for t in finals] == [[0.9], [0.5]]
    assert [t["meta"]["tree_chunk_dc"] for t in finals] == [[0.1], [0.4]]
    assert [t["meta"]["tree_selected"] for t in finals] == [True, True]
    assert result.case_info[0]["keep"] == "c"
    assert result.case_info[0]["round_log"][0]["n_extra"] == 0
    assert result.case_info[0]["n_final"] == 2


def test_keep_c_last_all_keeps_the_final_six():
    # Round 0 keeps the two higher C branches and drops branch 0.
    # Round 1 is the last expansion, so all six children stay for voting.
    table = {
        (0, 0, 0, 0): {"dc": 0.1, "c": 0.2},
        (0, 0, 0, 1): {"dc": 0.9, "c": 0.9},
        (0, 0, 0, 2): {"dc": 0.4, "c": 0.5},
        (0, 1, 0, 0): {"dc": 0.2, "c": 0.3},
        (0, 1, 0, 1): {"dc": 0.3, "c": 0.8},
        (0, 1, 0, 2): {"dc": 0.15, "c": 0.4},
        (0, 1, 1, 0): {"dc": 0.95, "c": 0.7},
        (0, 1, 1, 1): {"dc": 0.05, "c": 0.1},
        (0, 1, 1, 2): {"dc": 0.7, "c": 0.6},
    }
    generate, calls, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=1000, tree_chunk=500, tree_keep="c", tree_last_all=True),
        "qwen3",
        generate,
    )
    assert len([c for c in calls if c["round"] == 0]) == 3
    assert len([c for c in calls if c["round"] == 1]) == 6
    finals = result.bundles[0]
    assert len(finals) == 6
    assert sum(t["meta"]["tree_selected"] for t in finals) == 2
    assert result.case_info[0]["last_all"] is True
    assert result.case_info[0]["n_pruned"] == 1
    assert result.case_info[0]["round_log"][0]["last_round"] is False
    assert result.case_info[0]["round_log"][0]["n_extra"] == 0
    assert result.case_info[0]["round_log"][1]["last_round"] is True
    assert result.case_info[0]["round_log"][1]["n_extra"] == 4


def test_keep_c4dc_and_cguard_on_three_children():
    table = {
        (0, 0, 0, 0): {"dc": 0.9, "c": 0.2},
        (0, 0, 0, 1): {"dc": 0.1, "c": 0.9},
        (0, 0, 0, 2): {"dc": 0.4, "c": 0.5},
    }

    def paths(keep):
        generate, _, _ = _script(table)
        result = run_dc_chunk_search(
            None,
            _Tok(),
            ["prompt"],
            _args(max_new_tokens=500, tree_chunk=500, tree_keep=keep),
            "qwen3",
            generate,
        )
        return [t["meta"]["tree_branch_path"] for t in result.bundles[0]]

    # All three sit inside the C top-4, so the keep is the two highest ΔC.
    assert paths("c4dc") == [[0], [2]]
    # C rank 1 stays, and the other slot is the best ΔC among the rest.
    assert paths("cguard") == [[1], [0]]


def test_finished_trace_occupies_a_beam_slot():
    # C-only keeps the EOS child and one open child. The next round expands
    # only that open child, and only one slot is left.
    table = {
        (0, 0, 0, 0): {"dc": 0.1, "c": 0.2},
        (0, 0, 0, 1): {"dc": 0.2, "c": 0.9, "stop": True},
        (0, 0, 0, 2): {"dc": 0.3, "c": 0.5},
        (0, 1, 0, 0): {"dc": 0.4, "c": 0.2},
        (0, 1, 0, 1): {"dc": 0.5, "c": 0.3},
        (0, 1, 0, 2): {"dc": 0.1, "c": 0.8},
    }
    generate, calls, _ = _script(table)
    result = run_dc_chunk_search(
        None,
        _Tok(),
        ["prompt"],
        _args(max_new_tokens=1000, tree_chunk=500, tree_keep="c"),
        "qwen3",
        generate,
    )
    assert len([c for c in calls if c["round"] == 1]) == 3
    finals = result.bundles[0]
    assert len(finals) == 2
    assert [t["meta"]["tree_branch_path"] for t in finals] == [[1], [2, 2]]
    assert [t["finished"] for t in finals] == [True, False]
    assert result.case_info[0]["n_pruned"] == 1 + 2


def test_cuda_graph_budget_follows_the_expansion_width():
    args = _args(end_idx=20, start_idx=0, num_samples=1, max_batch=512, dp_size=1)
    # 20 problems * 2 * 3 = 120 concurrent chunks → the 160 graph bucket.
    assert _cuda_graph_max_bs(args) == 160
    args.tree_search = False
    assert _cuda_graph_max_bs(args) == 32
