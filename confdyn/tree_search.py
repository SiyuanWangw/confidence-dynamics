"""Chunk tree search.

One search per problem. Each live trajectory is expanded into ``branch``
continuations of at most ``chunk`` new tokens (or until EOS). Each round keeps
``beam`` chunks, not one child per parent. ``dc`` keeps the highest mean
penultimate-layer ΔC, ``c`` the highest mean final-layer C, and ``random``
draws that many chunks uniformly.

Round 1 starts from the prompt. It ranks ``tree_init_branch`` chunks when that
is set, otherwise ``branch`` (3 → 2 with the defaults). Later rounds expand
every still-open survivor by ``branch``, so a full beam ranks ``beam * branch``
chunks (6 → 2, or 4 → 2 when branch is 2). A selected chunk that already hit EOS or the
generation budget is stored and not extended. On an intermediate round, a
chunk outside the top ``beam`` is dropped, including an early EOS. The next
round expands only the selected chunks that are still open.

The last round for a problem is the expansion after which nothing is still
open. Every chunk from that round is kept, selected or not. Finished chunks
that were selected on earlier rounds stay in the pool as well.

``tree_keep`` replaces that rule. ``c`` keeps the highest chunk-C children.
``c4dc`` takes the top 4 by chunk C, then the highest chunk ΔC inside that
pool. ``cguard`` always keeps C rank 1 and fills the other slots from C ranks
2–4 by ΔC. ``c1dc`` ranks every child by chunk C + chunk ΔC, and ``c2dc``
by chunk C + 2·chunk ΔC. A finished
trajectory occupies one beam slot and is not expanded,
and the last round does not keep the unselected chunks, so a problem ends
with at most ``beam`` trajectories. ``tree_last_all`` keeps every candidate
from that last round, including the ones outside the beam, so they can be
voted. Earlier rounds still drop them. Budget-exhausted trajectories stay in
that pool and are scored incorrect by the caller.

Each chunk is a new SGLang request on the full token prefix. Presence penalty
therefore restarts on the new tokens; earlier chunks sit in the prompt and are
not in that penalty window.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from confdyn import filter_metrics as fm
from confdyn.engine import _generate_ids, _sampling_params
from confdyn.margins import _log_margin_presence
from confdyn.prompting import FAMILY_SAMPLING

# case stride stays above round * _SEED_ROUND for any context-length budget.
# round stride stays above parent_slot * branch_factor + branch.
_SEED_CASE = 1_000_000_000_000
_SEED_ROUND = 1_000_000

_META_SKIP = {
    "id",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "cached_tokens_details",
    "weight_version",
    "num_retractions",
    "reasoning_tokens",
    "dp_rank",
    "tree_chunk_dc",
    "tree_chunk_score",
    "tree_branch_path",
    "tree_score",
    "tree_selected",
    "tree_rank",
}

# Chunk ranking signal. ``c`` is the final-layer DeepConf C (``last_c``).
# ``random`` ignores the score and draws ``beam`` chunks without replacement.
_SCORE_KEY = {
    "dc": "penult_dc",
    "c": "last_c",
}
_SCORE_MODES = ("dc", "c", "random")


@dataclass
class _Node:
    input_ids: List[int]
    gen_ids: List[int]
    meta_parts: List[dict]
    chunk_scores: List[Optional[float]]
    branch_path: List[int]
    generated: int
    stopped: bool
    chunk_dc: Optional[float] = None
    chunk_c: Optional[float] = None
    chunk_c_scores: Optional[List[Optional[float]]] = None
    selected: bool = False


@dataclass
class _Job:
    case: int
    parent: _Node
    slot: int
    branch: int
    n_new: int
    seed: int


@dataclass
class TreeSearchResult:
    bundles: List[List[dict]]
    case_info: List[dict]
    expanded_tokens: int
    rounds: int


def expansion_seed(
    base: int,
    case_idx: int,
    round_idx: int,
    parent_slot: int,
    branch: int,
    branch_factor: int,
) -> int:
    """Unique sampling seed for one (problem, round, parent, branch)."""
    within = int(parent_slot) * int(branch_factor) + int(branch)
    if within < 0 or within >= _SEED_ROUND:
        raise ValueError(
            f"parent_slot={parent_slot} branch={branch} does not fit in one round"
        )
    return (
        int(base)
        + int(case_idx) * _SEED_CASE
        + int(round_idx) * _SEED_ROUND
        + within
    )


def parse_expansion_seed(seed: int, base: int, branch_factor: int):
    """Inverse of ``expansion_seed`` for tests and logs."""
    rel = int(seed) - int(base)
    case, rem = divmod(rel, _SEED_CASE)
    round_idx, within = divmod(rem, _SEED_ROUND)
    slot, branch = divmod(within, int(branch_factor))
    return case, round_idx, slot, branch


def select_top(children: Sequence[Any], beam_width: int) -> List[Any]:
    """Highest chunk score first. Missing scores lose. Ties keep earlier order."""
    width = max(int(beam_width), 0)

    def key(index: int):
        score = children[index].chunk_dc
        missing = score is None
        return (missing, 0.0 if missing else -float(score), index)

    order = sorted(range(len(children)), key=key)
    return [children[i] for i in order[:width]]


def _order_by(children: Sequence[Any], attr: str) -> List[int]:
    """Highest ``attr`` first. Missing scores lose. Ties keep earlier order."""

    def key(index: int):
        score = getattr(children[index], attr)
        missing = score is None
        return (missing, 0.0 if missing else -float(score), index)

    return sorted(range(len(children)), key=key)


def select_keep_c(children: Sequence[Any], beam_width: int) -> List[Any]:
    """Keep the children with the highest chunk C."""
    width = max(int(beam_width), 0)
    order = _order_by(children, "chunk_c")
    return [children[i] for i in order[:width]]


def select_keep_c4dc(children: Sequence[Any], beam_width: int, pool: int = 4) -> List[Any]:
    """Top ``pool`` by chunk C, then the highest chunk ΔC inside that pool."""
    width = max(int(beam_width), 0)
    by_c = _order_by(children, "chunk_c")
    pool_idx = by_c[: min(max(int(pool), 0), len(by_c))]

    def key(pos: int):
        score = children[pool_idx[pos]].chunk_dc
        missing = score is None
        return (missing, 0.0 if missing else -float(score), pos)

    ranked = sorted(range(len(pool_idx)), key=key)
    return [children[pool_idx[i]] for i in ranked[:width]]


def select_keep_c2dc(children: Sequence[Any], beam_width: int, alpha: float = 2.0) -> List[Any]:
    """Keep the highest chunk C + ``alpha``·chunk ΔC. Missing either score loses."""
    width = max(int(beam_width), 0)

    def key(index: int):
        child = children[index]
        c_score = child.chunk_c
        dc_score = child.chunk_dc
        missing = c_score is None or dc_score is None
        score = 0.0 if missing else float(c_score) + float(alpha) * float(dc_score)
        return (missing, -score, index)

    order = sorted(range(len(children)), key=key)
    return [children[i] for i in order[:width]]


def select_keep_cguard(children: Sequence[Any], beam_width: int, window: int = 4) -> List[Any]:
    """Keep C rank 1. Fill the other slots from C ranks 2..``window`` by ΔC."""
    width = max(int(beam_width), 0)
    if width <= 0 or not children:
        return []
    by_c = _order_by(children, "chunk_c")
    chosen = [by_c[0]]
    if width == 1:
        return [children[chosen[0]]]
    band = by_c[1:window]

    def key(pos: int):
        score = children[band[pos]].chunk_dc
        missing = score is None
        return (missing, 0.0 if missing else -float(score), pos)

    for pos in sorted(range(len(band)), key=key):
        if len(chosen) >= width:
            break
        chosen.append(band[pos])
    return [children[i] for i in chosen[:width]]


def select_random(children: Sequence[Any], beam_width: int, rng: random.Random) -> List[Any]:
    """Draw ``beam_width`` chunks without replacement. Scores are ignored."""
    width = max(int(beam_width), 0)
    n = len(children)
    if width >= n:
        return list(children)
    indices = rng.sample(range(n), width)
    return [children[i] for i in indices]


def selection_seed(base: int, case_idx: int, round_idx: int) -> int:
    """RNG seed for one random-selection round. Distinct from expansion seeds."""
    return (
        int(base)
        + int(case_idx) * _SEED_CASE
        + int(round_idx) * _SEED_ROUND
        + (_SEED_ROUND - 1)
    )


def tree_score_name(args) -> str:
    """``dc`` / ``c`` rank by a ΔC or C mean. ``random`` draws the beam uniformly."""
    name = str(getattr(args, "tree_score", "dc") or "dc").strip().lower()
    if name not in _SCORE_MODES:
        raise ValueError(f"tree_score must be dc, c, or random, got {name!r}")
    return name


_KEEP_MODES = ("score", "c", "c4dc", "cguard", "c1dc", "c2dc")


def tree_keep_name(args) -> str:
    """``score`` is the single-score beam. The other modes keep at most ``beam`` traces.

    ``c`` keeps the highest chunk C. ``c4dc`` takes the top 4 by C, then the
    highest ΔC. ``cguard`` keeps C rank 1 and picks the rest from C ranks 2–4
    by ΔC. ``c1dc`` keeps the highest chunk C + chunk ΔC. ``c2dc`` keeps the
    highest chunk C + 2·chunk ΔC. A finished trace occupies a slot and the
    last round drops the rest.
    """
    name = str(getattr(args, "tree_keep", "score") or "score").strip().lower()
    if name not in _KEEP_MODES:
        raise ValueError(
            f"tree_keep must be score, c, c4dc, cguard, c1dc, or c2dc, got {name!r}"
        )
    return name


def tree_rank_name(args) -> str:
    """``chunk`` scores the new tokens only. ``tail`` scores the running tail."""
    name = str(getattr(args, "tree_rank", "chunk") or "chunk").strip().lower()
    if name not in ("chunk", "tail"):
        raise ValueError(f"tree_rank must be chunk or tail, got {name!r}")
    return name


def _finite(score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _concat_steps(parts: Sequence[dict], key: str) -> List[Any]:
    steps: List[Any] = []
    for part in parts:
        values = part.get(key)
        if values:
            steps.extend(values)
    return steps


def _rank_score(
    parts: Sequence[dict],
    score_name: str,
    rank: str,
    tail_window: int,
) -> Optional[float]:
    """Score one expansion.

    ``chunk`` is the mean over the new tokens only. ``tail`` is the mean of
    the last ``tail_window`` steps of the whole generation so far. Random
    selection still records the new chunk's mean ΔC and does not use it.
    """
    if score_name == "random":
        newest = parts[-1] if parts else {}
        return _finite(fm.mean_steps(newest.get("penult_dc")))
    key = _SCORE_KEY[score_name]
    if rank == "tail":
        return _finite(fm.mean_tail_steps(_concat_steps(parts, key), tail_window))
    newest = parts[-1] if parts else {}
    return _finite(fm.mean_steps(newest.get(key)))


def _chunk_stopped(meta: dict, n_new: int, n_out: int) -> bool:
    if n_out <= 0:
        return True
    reason = meta.get("finish_reason")
    kind = None
    if isinstance(reason, dict):
        kind = reason.get("type")
    elif reason not in (None, "", {}):
        kind = str(reason)
    if kind in ("stop", "eos"):
        return True
    if kind == "length":
        return False
    return n_out < int(n_new)


def _output_ids(out: dict) -> List[int]:
    raw = out.get("output_ids") or []
    return [int(x) for x in raw]


def _merge_metas(parts: Sequence[dict], generated: int, stopped: bool) -> dict:
    merged: Dict[str, Any] = {
        "completion_tokens": int(generated),
        "finish_reason": {
            "type": "stop" if stopped else "length",
            "length": int(generated),
        },
    }
    keys = set()
    for part in parts:
        keys.update(part.keys())
    for key in keys:
        if key in _META_SKIP:
            continue
        vals = [part.get(key) for part in parts]
        if not all(v is None or isinstance(v, list) for v in vals):
            continue
        acc: List[Any] = []
        for value in vals:
            if value:
                acc.extend(value)
        if acc:
            merged[key] = acc
    return merged


def _encode_prompt(tokenizer, prompt: str) -> List[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    return [int(x) for x in ids]


def run_dc_chunk_search(
    llm,
    tokenizer,
    prompts: Sequence[str],
    args,
    family: str,
    generate_fn: Optional[Callable] = None,
) -> TreeSearchResult:
    """Expand ``prompts`` with ΔC chunk selection. One tree per prompt."""
    beam = max(int(getattr(args, "tree_beam", 2) or 2), 1)
    branch = max(int(getattr(args, "tree_branch", 3) or 3), 1)
    init_branch = max(int(getattr(args, "tree_init_branch", 0) or 0), 0)
    first_branch = init_branch if init_branch > 0 else branch
    seed_factor = max(first_branch, branch)
    chunk = max(int(getattr(args, "tree_chunk", 500) or 500), 1)
    warmup = max(int(getattr(args, "tree_warmup", 0) or 0), 0)
    score_name = tree_score_name(args)
    keep_name = tree_keep_name(args)
    hold = keep_name != "score"
    rank = tree_rank_name(args)
    tail_window = max(int(getattr(args, "tree_tail", 2048) or 2048), 1)
    max_new = max(int(getattr(args, "max_new_tokens", 1) or 1), 1)
    max_batch = max(int(getattr(args, "max_batch", 1) or 1), 1)
    base_seed = int(getattr(args, "random_seed", 0) or 0)
    case0 = int(getattr(args, "start_idx", 0) or 0)
    skip_special = bool(FAMILY_SAMPLING.get(family, {}).get("skip_special_tokens", True))

    if generate_fn is None:
        def generate_fn(input_ids, sampling_params):
            return _generate_ids(llm, input_ids, sampling_params, args)

    n_cases = len(prompts)
    active: List[List[_Node]] = []
    for prompt in prompts:
        ids = _encode_prompt(tokenizer, prompt)
        if not ids:
            raise RuntimeError("tree search received an empty prompt")
        active.append(
            [
                _Node(
                    input_ids=ids,
                    gen_ids=[],
                    meta_parts=[],
                    chunk_scores=[],
                    branch_path=[],
                    generated=0,
                    stopped=False,
                )
            ]
        )
    completed: List[List[_Node]] = [[] for _ in range(n_cases)]
    expanded = [0 for _ in range(n_cases)]
    pruned = [0 for _ in range(n_cases)]
    n_held = [0 for _ in range(n_cases)]
    case_rounds = [0 for _ in range(n_cases)]
    round_logs: List[List[dict]] = [[] for _ in range(n_cases)]

    round_idx = 0
    logged_margins = False
    expanded_tokens = 0
    print(
        f"[tree] score={score_name} keep={keep_name} rank={rank} tail={tail_window} "
        f"beam={beam} branch={branch} init={first_branch} chunk={chunk} warmup={warmup} "
        f"max_new_tokens={max_new} cases={n_cases}  "
        f"round1={first_branch}→{beam}  later≤{beam * branch}→{beam}",
        flush=True,
    )

    while any(active):
        t0 = time.perf_counter()
        jobs: List[_Job] = []
        for case, beams in enumerate(active):
            if hold and n_held[case] >= beam:
                for node in beams:
                    completed[case].append(node)
                continue
            for slot, node in enumerate(beams):
                remain = max_new - node.generated
                if node.stopped or remain <= 0:
                    completed[case].append(node)
                    continue
                length = warmup if round_idx == 0 and warmup > 0 else chunk
                n_new = min(length, remain)
                round_branch = first_branch if round_idx == 0 else branch
                for b in range(round_branch):
                    jobs.append(
                        _Job(
                            case=case,
                            parent=node,
                            slot=slot,
                            branch=b,
                            n_new=n_new,
                            seed=expansion_seed(
                                base_seed,
                                case0 + case,
                                round_idx,
                                slot,
                                b,
                                seed_factor,
                            ),
                        )
                    )
        active = [[] for _ in range(n_cases)]
        if not jobs:
            break

        outputs: List[dict] = []
        for start in range(0, len(jobs), max_batch):
            batch = jobs[start : start + max_batch]
            sps = []
            for job in batch:
                sp = _sampling_params(args, family, sampling_seed=job.seed)
                sp["max_new_tokens"] = job.n_new
                sp["n"] = 1
                sps.append(sp)
            outs = generate_fn([job.parent.input_ids for job in batch], sps)
            if len(outs) != len(batch):
                raise RuntimeError(
                    f"tree search generate returned {len(outs)} outputs for {len(batch)} chunks"
                )
            outputs.extend(outs)

        if not logged_margins:
            _log_margin_presence(
                [o.get("meta_info") or {} for o in outputs],
                family,
            )
            logged_margins = True

        children: List[List[_Node]] = [[] for _ in range(n_cases)]
        for job, out in zip(jobs, outputs):
            meta = out.get("meta_info") or {}
            new_ids = _output_ids(out)
            parent = job.parent
            parts = parent.meta_parts + [meta]
            if hold:
                dc_score = _rank_score(parts, "dc", rank, tail_window)
                c_score = _rank_score(parts, "c", rank, tail_window)
                score = dc_score
                c_scores = list(parent.chunk_c_scores or []) + [c_score]
            else:
                dc_score = None
                c_score = None
                score = _rank_score(parts, score_name, rank, tail_window)
                c_scores = None
            stopped = _chunk_stopped(meta, job.n_new, len(new_ids))
            child = _Node(
                input_ids=parent.input_ids + new_ids,
                gen_ids=parent.gen_ids + new_ids,
                meta_parts=parts,
                chunk_scores=parent.chunk_scores + [score],
                branch_path=parent.branch_path + [job.branch],
                generated=parent.generated + len(new_ids),
                stopped=stopped,
                chunk_dc=score,
                chunk_c=c_score,
                chunk_c_scores=c_scores,
            )
            expanded[job.case] += len(new_ids)
            expanded_tokens += len(new_ids)
            children[job.case].append(child)

        n_open = 0
        n_done = 0
        n_pruned = 0
        for case, kids in enumerate(children):
            if not kids:
                continue
            case_rounds[case] = round_idx + 1
            width = beam - n_held[case] if hold else beam
            if width <= 0:
                pruned[case] += len(kids)
                n_pruned += len(kids)
                continue
            if keep_name == "c":
                picked = select_keep_c(kids, width)
            elif keep_name == "c4dc":
                picked = select_keep_c4dc(kids, width)
            elif keep_name == "cguard":
                picked = select_keep_cguard(kids, width)
            elif keep_name == "c1dc":
                picked = select_keep_c2dc(kids, width, alpha=1.0)
            elif keep_name == "c2dc":
                picked = select_keep_c2dc(kids, width)
            elif score_name == "random":
                rng = random.Random(
                    selection_seed(base_seed, case0 + case, round_idx)
                )
                picked = select_random(kids, width, rng)
            else:
                picked = select_top(kids, width)
            picked_ids = {id(child) for child in picked}
            n_done_case = 0
            for child in picked:
                child.selected = True
                # A selected EOS/cap chunk occupies its slot: it is stored and
                # not replaced by the next-best open chunk.
                if child.stopped or child.generated >= max_new:
                    completed[case].append(child)
                    n_done += 1
                    n_done_case += 1
                    if hold:
                        n_held[case] += 1
                else:
                    active[case].append(child)
                    n_open += 1
            # Nothing left to extend: this is the last round, so keep every
            # candidate. Earlier rounds still drop chunks outside the beam.
            # A keep policy drops those extras unless tree_last_all is set,
            # which stores them for voting and does not extend them.
            last_round = not active[case]
            keep_last = bool(getattr(args, "tree_last_all", False))
            n_extra = 0
            if hold and not (keep_last and last_round):
                n_drop = len(kids) - len(picked)
                pruned[case] += n_drop
                n_pruned += n_drop
            elif last_round:
                for child in kids:
                    if id(child) in picked_ids:
                        continue
                    completed[case].append(child)
                    n_done += 1
                    n_done_case += 1
                    n_extra += 1
            else:
                n_drop = len(kids) - len(picked)
                pruned[case] += n_drop
                n_pruned += n_drop
            round_logs[case].append(
                {
                    "round": round_idx + 1,
                    "n_cand": len(kids),
                    "selected_dc": [child.chunk_dc for child in picked],
                    "selected_c": [child.chunk_c for child in picked],
                    "n_done": n_done_case,
                    "last_round": last_round,
                    "n_extra": n_extra,
                }
            )

        dt = time.perf_counter() - t0
        print(
            f"[tree] round {round_idx + 1}  chunks={len(jobs)}  "
            f"kept_open={n_open}  kept_done={n_done}  pruned={n_pruned}  "
            f"tokens={expanded_tokens}  {dt:.1f}s",
            flush=True,
        )
        round_idx += 1

    bundles: List[List[dict]] = []
    case_info: List[dict] = []
    for case, nodes in enumerate(completed):
        traces = []
        for node in nodes:
            text = tokenizer.decode(node.gen_ids, skip_special_tokens=skip_special)
            meta = _merge_metas(node.meta_parts, node.generated, node.stopped)
            rounded_scores = [
                None if score is None else round(float(score), 6)
                for score in node.chunk_scores
            ]
            meta["tree_score"] = score_name
            meta["tree_keep"] = keep_name
            meta["tree_rank"] = rank
            meta["tree_chunk_score"] = rounded_scores
            meta["tree_chunk_dc"] = rounded_scores
            if node.chunk_c_scores:
                meta["tree_chunk_c"] = [
                    None if score is None else round(float(score), 6)
                    for score in node.chunk_c_scores
                ]
            meta["tree_branch_path"] = list(node.branch_path)
            meta["tree_selected"] = bool(node.selected)
            traces.append(
                {
                    "text": text,
                    "meta": meta,
                    "completion_tokens": node.generated,
                    "finished": bool(node.stopped),
                }
            )
        bundles.append(traces)
        case_info.append(
            {
                "score": score_name,
                "keep": keep_name,
                "rank": rank,
                "tail": tail_window,
                "warmup": warmup,
                "beam": beam,
                "branch": branch,
                "init_branch": first_branch,
                "last_all": bool(getattr(args, "tree_last_all", False)),
                "chunk": chunk,
                "rounds": case_rounds[case],
                "expanded_tokens": expanded[case],
                "n_final": len(traces),
                "n_pruned": pruned[case],
                "round_log": round_logs[case],
            }
        )
    return TreeSearchResult(
        bundles=bundles,
        case_info=case_info,
        expanded_tokens=expanded_tokens,
        rounds=round_idx,
    )
