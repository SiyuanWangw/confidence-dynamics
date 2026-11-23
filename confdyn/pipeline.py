"""Dataset over-generate → filter / vote evaluation loop."""

from __future__ import annotations

import os
import time
from typing import List, Optional

from confdyn import aggregation as agg
from confdyn import filter_metrics as fm
from confdyn.data import (
    MATH_QA_DATASETS,
    MCQ_DATASETS,
    VLM_MCQ_DATASETS,
    _case_record,
    _chat_prompt_and_images,
    _dataset_defaults,
    _judge_sample,
    _load_samples,
)
from confdyn.dump_io import _dump_step_stats_enabled, _fmt_hms
from confdyn.engine import (
    _expand_sample_major,
    _expand_sample_major_images,
    _generate_chunk,
    _sampling_params,
    _sampling_params_for_traces,
    _trace_seed,
)
from confdyn.margins import (
    _approx_entropy,
    _diversity_summary,
    _log_margin_presence,
    _mean_logprob,
    _meta_means,
)
from confdyn.prompting import (
    _think_answer_lens,
    _think_end_token,
)
from confdyn.tree_search import run_dc_chunk_search

_STAT_PRINT_KEYS = (
    "total_num",
    "pass@1",
    "pass@k",
    "vote_acc",
    "vote_acc_best_tail_dc",
    "vote_acc_best_dc",
    "vote_acc_majority",
    "vote_acc_tail_dc_weighted",
    "vote_acc_dc_weighted",
    "vote_acc_tail_c_weighted",
    "vote_acc_tail_c_filter_30",
    "pass@1_tail_c_filter_30",
    "vote_acc_tail_c_filter_50",
    "pass@1_tail_c_filter_50",
    "vote_acc_joint_dc_weighted_50",
    "vote_acc_joint_dc_weighted_60",
    "pass@1_joint_50",
    "pass@1_joint_60",
    "mean_joint_retained_50",
    "mean_joint_retained_60",
    "joint_rho",
    "tail_mode",
    "mean_tail_c_correct",
    "mean_tail_c_incorrect",
    "mean_tail_dc_correct",
    "mean_tail_dc_incorrect",
    "avg_token_length-all",
    "mean_unique_seq_ratio",
    "generate_s",
    "judge_s",
    "wall_s",
)


def run_eval_dataset(args, tokenizer, llm, family: str) -> dict:
    dataset = args.dataset
    data_path = os.path.abspath(args.data_path or _dataset_defaults(dataset))
    samples = _load_samples(data_path)
    start = args.start_idx
    end = min(args.end_idx, len(samples))
    subset = samples[start:end]
    n = args.num_samples
    fk = max(int(args.filter_k), 1)
    tree = bool(getattr(args, "tree_search", False))
    if tree:
        print(
            f"{dataset}: {data_path}  cases [{start}, {end}) = {len(subset)}  "
            f"tree beam={int(args.tree_beam)} branch={int(args.tree_branch)} "
            f"init={int(getattr(args, 'tree_init_branch', 0) or 0)} "
            f"chunk={int(args.tree_chunk)} warmup={int(getattr(args, 'tree_warmup', 0) or 0)} "
            f"score={getattr(args, 'tree_score', 'dc')} "
            f"rank={getattr(args, 'tree_rank', 'chunk')}  "
            f"filter_k={fk}"
        )
    else:
        print(f"{dataset}: {data_path}  cases [{start}, {end}) = {len(subset)}  N={n}  filter_k={fk}")
    _jr = (
        float(args.joint_rho)
        if getattr(args, "joint_rho", None) is not None
        else agg.default_rho(family)
    )
    _tm, _tv = agg.resolve_tail_mode(family)
    print(
        f"  paper: joint_rho={_jr:.2f}  tail={_tm}:{_tv:g}  "
        f"baselines=MV / TailC-W / TailC-F@{{20,30,50}}% / Ours(C∩ΔC+ΔC-W)",
        flush=True,
    )

    case_prompts = []
    case_images = []
    for sample in subset:
        prompt, images = _chat_prompt_and_images(dataset, sample, tokenizer, family)
        case_prompts.append(prompt)
        case_images.append(images)
    n_img = sum(1 for imgs in case_images if imgs)
    t_wall0 = time.perf_counter()
    t_gen0 = time.perf_counter()
    search_info = None
    if tree:
        if n_img:
            raise RuntimeError("ΔC chunk tree search does not support image prompts")
        print(
            f"Tree search {len(case_prompts)} problems: {_sampling_params(args, family)} "
            f"dp={max(int(getattr(args, 'dp_size', 1) or 1), 1)}",
            flush=True,
        )
        searched = run_dc_chunk_search(llm, tokenizer, case_prompts, args, family)
        bundles = searched.bundles
        search_info = searched.case_info
        generated_token_total = int(searched.expanded_tokens)
    else:
        # Sample-major so DP=2/4 round-robin does not pair clones of the same prompt.
        prompts, dest = _expand_sample_major(case_prompts, n)
        image_data = _expand_sample_major_images(case_images, n)
        seed0 = _trace_seed(args, start, 0, n)
        seed1 = _trace_seed(args, start + max(len(subset) - 1, 0), max(n - 1, 0), n)
        print(
            f"Generating {len(prompts)} sequences: {_sampling_params(args, family)} "
            f"dp={max(int(getattr(args, 'dp_size', 1) or 1), 1)} "
            f"layout=sample-major sampling_seed={seed0}..{seed1}"
            + (f"  multimodal_cases={n_img}" if n_img else ""),
            flush=True,
        )

        n_out = len(prompts)
        texts: List[Optional[str]] = [None] * n_out
        token_counts: List[Optional[int]] = [None] * n_out
        finished: List[Optional[bool]] = [None] * n_out
        metas: List[Optional[dict]] = [None] * n_out
        batch = max(int(args.max_batch), 1)
        i = 0
        first_chunk = True
        while i < len(prompts):
            chunk = prompts[i : i + batch]
            chunk_dest = dest[i : i + batch]
            print(f"  batch {i}:{i + len(chunk)} / {len(prompts)}", flush=True)
            sps = _sampling_params_for_traces(args, family, chunk_dest, n, start)
            chunk_images = image_data[i : i + batch] if image_data is not None else None
            outs = _generate_chunk(llm, chunk, sps, args, image_data=chunk_images)
            if len(outs) != len(chunk):
                raise RuntimeError(
                    f"generate returned {len(outs)} outputs for {len(chunk)} prompts"
                )
            for o, d in zip(outs, chunk_dest):
                texts[d] = o.get("text", "")
                meta = o.get("meta_info") or {}
                metas[d] = meta
                token_counts[d] = int(meta.get("completion_tokens") or 0)
                reason = meta.get("finish_reason")
                if isinstance(reason, dict) and reason.get("type"):
                    finished[d] = reason.get("type") == "stop"
                elif reason in (None, "", {}):
                    finished[d] = None
                else:
                    finished[d] = str(reason) in ("stop", "eos", "True")
            i += len(chunk)
            if first_chunk:
                _log_margin_presence([m for m in metas if m is not None], family)
                first_chunk = False
        if any(t is None for t in texts):
            raise RuntimeError("sample-major scatter missed some traces")
        bundles = []
        for k in range(len(subset)):
            base = k * n
            bundles.append(
                [
                    {
                        "text": texts[base + j],
                        "meta": metas[base + j] or {},
                        "completion_tokens": token_counts[base + j],
                        "finished": finished[base + j],
                    }
                    for j in range(n)
                ]
            )
        generated_token_total = sum(int(x) for x in token_counts if x is not None)
    generate_s = time.perf_counter() - t_gen0
    t_judge0 = time.perf_counter()

    results = []

    def _append_case(row, case_i):
        if search_info is not None:
            row["tree"] = search_info[case_i]
        results.append(row)

    for k, sample in enumerate(subset):
        bundle = bundles[k]
        case_texts = [b["text"] for b in bundle]
        case_toks = [b["completion_tokens"] for b in bundle]
        case_fin = [b["finished"] for b in bundle]
        case_meta = [b["meta"] or {} for b in bundle]
        n_case = len(bundle)
        pass_list = []
        extracted = []
        judge_info = []
        token_id_lists = []
        # Keep policies store a budget-exhausted trace as unsolved. GPQA's
        # letter extract ignores finish_generation, so a capped trace that
        # happens to contain the right letter would otherwise count.
        hold_keep = str(getattr(args, "tree_keep", "score") or "score").strip().lower() not in (
            "",
            "score",
        )
        for text, fin in zip(case_texts, case_fin):
            try:
                ok, extracted_answer = _judge_sample(dataset, text, sample, fin, family)
            except Exception as exc:
                print(f"judge error idx={start + k}: {exc}", flush=True)
                ok, extracted_answer = False, "No extracted answer"
            unfinished = hold_keep and fin is False
            if unfinished:
                ok = False
            pass_list.append(1.0 if ok else 0.0)
            extracted.append(extracted_answer)
            judge_info.append(
                {
                    "rule_judge_result": bool(ok),
                    "finally_judge_result": bool(ok),
                    "extracted_answer": extracted_answer,
                    "fail_mode": "unfinished" if unfinished else ("ok" if ok else "fail"),
                }
            )
            token_id_lists.append(tokenizer.encode(text))
        think_lens = []
        ans_lens = []
        for text in case_texts:
            t, a = _think_answer_lens(text, tokenizer, _think_end_token(family))
            think_lens.append(t)
            ans_lens.append(a)
        max_new = int(getattr(args, "max_new_tokens", 0) or 0)
        finished_flags = [None if x is None else bool(x) for x in case_fin]
        finished_keep = fm.untruncated_indices(finished_flags, case_toks, max_new)
        # Length-capped tree beams stay in the vote, and n_truncated stays 0
        # for that path. keep_truncated is ordinary N sampling: every trace
        # stays, and n_truncated counts the over-length ones that were kept.
        if tree:
            keep = list(range(n_case))
            n_trunc = 0
        elif bool(getattr(args, "keep_truncated", False)):
            keep = list(range(n_case))
            n_trunc = n_case - len(finished_keep)
        else:
            keep = finished_keep
            n_trunc = n_case - len(keep)
        n_kept = len(keep)

        def _take(seq):
            return [seq[i] for i in keep]

        if n_kept == 0:
            row = _case_record(dataset, sample)
            row.update(
                {
                    "idx": start + k,
                    "n": n_case,
                    "n_kept": 0,
                    "n_truncated": n_trunc,
                    "kept_indices": [],
                    "finished": finished_flags,
                    "passat1_list": [],
                    "paper_tail_c_list": [],
                    "paper_tail_dc_list": [],
                    "generated_tokens": [],
                    "mean_logprob_list": [],
                    "mean_entropy_list": [],
                    "diversity": {"unique_ratio": 0.0},
                }
            )
            _append_case(row, k)
            continue
        case_texts = _take(case_texts)
        case_toks = _take(case_toks)
        case_meta = _take(case_meta)
        pass_list = _take(pass_list)
        extracted = _take(extracted)
        judge_info = _take(judge_info)
        token_id_lists = _take(token_id_lists)
        think_lens = _take(think_lens)
        ans_lens = _take(ans_lens)
        if (
            dataset in MATH_QA_DATASETS
            or dataset in MCQ_DATASETS
            or dataset in VLM_MCQ_DATASETS
        ):
            vote_keys = extracted
        else:
            vote_keys = [bool(x) for x in pass_list]
        voted, first_index, vote_counts = fm.majority_vote(vote_keys)
        logps = [_mean_logprob(m) for m in case_meta]
        ents = [_approx_entropy(m) for m in case_meta]
        ents_f = [0.0 if e is None else float(e) for e in ents]
        logps_f = [float("-inf") if x is None else float(x) for x in logps]
        weights = fm.confidence_weights(logps, ents_f)
        lo_ent_idx = fm.topk_indices(ents_f, fk, highest=False)
        hi_logp_idx = fm.topk_indices(logps_f, fk, highest=True)
        p1_hi_ent, pk_hi_ent, _, _ = fm.split_by_score(pass_list, ents_f, fk, True)
        p1_lo_ent, pk_lo_ent, _, _ = fm.split_by_score(pass_list, ents_f, fk, False)
        p1_hi_lp, pk_hi_lp, _, _ = fm.split_by_score(pass_list, logps_f, fk, True)
        margin_rows = [_meta_means(m) for m in case_meta]
        last_mps = [r["mean_last_mp"] for r in margin_rows]
        last_mcs = [r["mean_last_mcos"] for r in margin_rows]
        last_cs = [r["mean_last_c"] for r in margin_rows]
        all_dmps = [r["mean_all_dmp"] for r in margin_rows]
        all_dmcs = [r["mean_all_dmc"] for r in margin_rows]
        all_dcs = [r["mean_all_dc"] for r in margin_rows]
        all_dcs_n3 = [r["mean_all_dc_n3"] for r in margin_rows]
        all_dcs_n4 = [r["mean_all_dc_n4"] for r in margin_rows]
        switch_dmps = [r["mean_switch_dmp"] for r in margin_rows]
        switch_dmcs = [r["mean_switch_dmc"] for r in margin_rows]
        match_dmps = [r["mean_match_dmp"] for r in margin_rows]
        match_dmcs = [r["mean_match_dmc"] for r in margin_rows]
        tail_cs = [r["mean_tail_c"] for r in margin_rows]
        tail_dmps = [r["mean_tail_dmp"] for r in margin_rows]
        tail_dmcs = [r["mean_tail_dmc"] for r in margin_rows]
        tail_dcs = [r["mean_tail_dc"] for r in margin_rows]
        tail_dcs_n3 = [r["mean_tail_dc_n3"] for r in margin_rows]
        tail_dcs_n4 = [r["mean_tail_dc_n4"] for r in margin_rows]
        tail10_cs = [r["mean_tail10_c"] for r in margin_rows]
        tail10_dmps = [r["mean_tail10_dmp"] for r in margin_rows]
        tail10_dmcs = [r["mean_tail10_dmc"] for r in margin_rows]
        tail10_dcs = [r["mean_tail10_dc"] for r in margin_rows]
        tail10_dcs_n3 = [r["mean_tail10_dc_n3"] for r in margin_rows]
        tail10_dcs_n4 = [r["mean_tail10_dc_n4"] for r in margin_rows]
        tail20_cs = [r["mean_tail20_c"] for r in margin_rows]
        tail20_dmps = [r["mean_tail20_dmp"] for r in margin_rows]
        tail20_dmcs = [r["mean_tail20_dmc"] for r in margin_rows]
        tail20_dcs = [r["mean_tail20_dc"] for r in margin_rows]
        tail20_dcs_n3 = [r["mean_tail20_dc_n3"] for r in margin_rows]
        tail20_dcs_n4 = [r["mean_tail20_dc_n4"] for r in margin_rows]

        # Paper signals: Qwen → last-2048 tail; Gemma → last-20% tail.
        tail_mode, _tail_val = agg.resolve_tail_mode(family)
        if tail_mode == "frac":
            paper_c, paper_dc = tail20_cs, tail20_dcs
        else:
            paper_c, paper_dc = tail_cs, tail_dcs
        joint_rho = (
            float(args.joint_rho)
            if getattr(args, "joint_rho", None) is not None
            else agg.default_rho(family)
        )
        paper = agg.evaluate_paper_methods(
            pass_list,
            vote_keys,
            paper_c,
            paper_dc,
            rho=joint_rho,
        )
        # Same retained pool. Tail ΔC is the paper tail (Qwen: last 2048
        # steps; Gemma: last 20%). ΔC is the mean over the whole trajectory.
        selected = agg.answer_selection(pass_list, vote_keys, paper_dc, all_dcs)
        best_tail_c_i = agg.best_score_index(paper_c)
        if best_tail_c_i is None:
            best_tail_c_i = 0
        vote_acc_best_tail_c = float(pass_list[best_tail_c_i]) if pass_list else 0.0

        def _sel(scores, highest=True):
            if not scores or all(s is None for s in scores):
                return None, None
            p1, pk, _, _ = fm.split_by_score(pass_list, scores, fk, highest)
            return p1, pk

        p1_hi_lmp, pk_hi_lmp = _sel(last_mps, True)
        p1_hi_lmc, pk_hi_lmc = _sel(last_mcs, True)
        p1_hi_lc, pk_hi_lc = _sel(last_cs, True)
        p1_hi_admp, pk_hi_admp = _sel(all_dmps, True)
        p1_hi_admc, pk_hi_admc = _sel(all_dmcs, True)
        p1_hi_adc, pk_hi_adc = _sel(all_dcs, True)
        p1_hi_adc_n3, pk_hi_adc_n3 = _sel(all_dcs_n3, True)
        p1_hi_adc_n4, pk_hi_adc_n4 = _sel(all_dcs_n4, True)
        p1_hi_sdmp, pk_hi_sdmp = _sel(switch_dmps, True)
        p1_hi_sdmc, pk_hi_sdmc = _sel(switch_dmcs, True)
        p1_hi_tc, pk_hi_tc = _sel(tail_cs, True)
        p1_hi_tdmp, pk_hi_tdmp = _sel(tail_dmps, True)
        p1_hi_tdmc, pk_hi_tdmc = _sel(tail_dmcs, True)
        p1_hi_tdc, pk_hi_tdc = _sel(tail_dcs, True)
        p1_hi_tdc_n3, pk_hi_tdc_n3 = _sel(tail_dcs_n3, True)
        p1_hi_tdc_n4, pk_hi_tdc_n4 = _sel(tail_dcs_n4, True)
        p1_hi_t10c, pk_hi_t10c = _sel(tail10_cs, True)
        p1_hi_t10dmp, pk_hi_t10dmp = _sel(tail10_dmps, True)
        p1_hi_t10dmc, pk_hi_t10dmc = _sel(tail10_dmcs, True)
        p1_hi_t10dc, pk_hi_t10dc = _sel(tail10_dcs, True)
        p1_hi_t10dc_n3, pk_hi_t10dc_n3 = _sel(tail10_dcs_n3, True)
        p1_hi_t10dc_n4, pk_hi_t10dc_n4 = _sel(tail10_dcs_n4, True)
        p1_hi_t20c, pk_hi_t20c = _sel(tail20_cs, True)
        p1_hi_t20dmp, pk_hi_t20dmp = _sel(tail20_dmps, True)
        p1_hi_t20dmc, pk_hi_t20dmc = _sel(tail20_dmcs, True)
        p1_hi_t20dc, pk_hi_t20dc = _sel(tail20_dcs, True)
        p1_hi_t20dc_n3, pk_hi_t20dc_n3 = _sel(tail20_dcs_n3, True)
        p1_hi_t20dc_n4, pk_hi_t20dc_n4 = _sel(tail20_dcs_n4, True)
        dump_steps = _dump_step_stats_enabled()
        sample_deltas = []
        for j, meta in enumerate(case_meta):
            rec = {
                "sample": j,
                "correct": bool(pass_list[j] >= 0.5) if j < len(pass_list) else False,
                "mean_last_mp": last_mps[j] if j < len(last_mps) else None,
                "mean_last_mcos": last_mcs[j] if j < len(last_mcs) else None,
                "mean_last_c": last_cs[j] if j < len(last_cs) else None,
                "mean_all_dmp": all_dmps[j] if j < len(all_dmps) else None,
                "mean_all_dmc": all_dmcs[j] if j < len(all_dmcs) else None,
                "mean_all_dc": all_dcs[j] if j < len(all_dcs) else None,
                "mean_all_dc_n3": all_dcs_n3[j] if j < len(all_dcs_n3) else None,
                "mean_all_dc_n4": all_dcs_n4[j] if j < len(all_dcs_n4) else None,
                "mean_match_dmp": match_dmps[j] if j < len(match_dmps) else None,
                "mean_match_dmc": match_dmcs[j] if j < len(match_dmcs) else None,
                "mean_switch_dmp": switch_dmps[j] if j < len(switch_dmps) else None,
                "mean_switch_dmc": switch_dmcs[j] if j < len(switch_dmcs) else None,
                "mean_tail_c": tail_cs[j] if j < len(tail_cs) else None,
                "mean_tail_dmp": tail_dmps[j] if j < len(tail_dmps) else None,
                "mean_tail_dmc": tail_dmcs[j] if j < len(tail_dmcs) else None,
                "mean_tail_dc": tail_dcs[j] if j < len(tail_dcs) else None,
                "mean_tail_dc_n3": tail_dcs_n3[j] if j < len(tail_dcs_n3) else None,
                "mean_tail_dc_n4": tail_dcs_n4[j] if j < len(tail_dcs_n4) else None,
                "mean_tail10_c": tail10_cs[j] if j < len(tail10_cs) else None,
                "mean_tail10_dmp": tail10_dmps[j] if j < len(tail10_dmps) else None,
                "mean_tail10_dmc": tail10_dmcs[j] if j < len(tail10_dmcs) else None,
                "mean_tail10_dc": tail10_dcs[j] if j < len(tail10_dcs) else None,
                "mean_tail10_dc_n3": tail10_dcs_n3[j] if j < len(tail10_dcs_n3) else None,
                "mean_tail10_dc_n4": tail10_dcs_n4[j] if j < len(tail10_dcs_n4) else None,
                "mean_tail20_c": tail20_cs[j] if j < len(tail20_cs) else None,
                "mean_tail20_dmp": tail20_dmps[j] if j < len(tail20_dmps) else None,
                "mean_tail20_dmc": tail20_dmcs[j] if j < len(tail20_dmcs) else None,
                "mean_tail20_dc": tail20_dcs[j] if j < len(tail20_dcs) else None,
                "mean_tail20_dc_n3": tail20_dcs_n3[j] if j < len(tail20_dcs_n3) else None,
                "mean_tail20_dc_n4": tail20_dcs_n4[j] if j < len(tail20_dcs_n4) else None,
            }
            if meta.get("tree_chunk_dc") is not None:
                rec["tree_score"] = meta.get("tree_score")
                rec["tree_keep"] = meta.get("tree_keep")
                rec["tree_chunk_score"] = meta.get("tree_chunk_score")
                rec["tree_chunk_dc"] = meta.get("tree_chunk_dc")
                if meta.get("tree_chunk_c") is not None:
                    rec["tree_chunk_c"] = meta.get("tree_chunk_c")
                rec["tree_branch_path"] = meta.get("tree_branch_path")
                rec["tree_selected"] = meta.get("tree_selected")
            if dump_steps:
                # table drops only the top-20 logit vectors. C, ΔC, match,
                # overlap, candidates, prob, and cos θ stay.
                table_dump = os.environ.get("OGF_LAYER_DUMP", "").strip().lower() == "table"
                rec["step_c"] = fm.step_values(meta.get("last_c"))
                rec["step_dc"] = fm.step_values(meta.get("penult_dc"))
                rec["step_dc_n3"] = fm.step_values(meta.get("n3_dc"))
                rec["step_dc_n4"] = fm.step_values(meta.get("n4_dc"))
                rec["step_match"] = fm.step_match_flags(meta.get("penult_match"))
                # log p_L(v*) and log p_{L-1}(v*). Their difference is Δlog p.
                rec["step_top1_logp"] = fm.step_values(meta.get("top1_logp"))
                rec["step_penult_top1_logp"] = fm.step_values(meta.get("penult_top1_logp"))
                if meta.get("last_rank_logp"):
                    rec["step_last_rank_logp"] = fm.step_rank_logps(meta, "last_rank_logp")
                    rec["step_penult_rank_logp"] = fm.step_rank_logps(meta, "penult_rank_logp")
                    rec["step_penult_on_last_logp"] = fm.step_rank_logps(
                        meta, "penult_on_last_logp"
                    )
                # Always keep the decoded id. layer_top_* dumps stay behind
                # OGF_LAYER_TOPK; they are much larger.
                rec["step_sampled_token"] = fm.step_sampled_tokens(meta)
                layer_depth = 1 + max(
                    (k for k in range(1, 10) if meta.get(f"layer_c_{k}")),
                    default=0,
                )
                if layer_depth > 1:
                    rec["step_layer_c"] = fm.step_layer_confidences(meta, layer_depth)
                if any(meta.get(f"layer_top_{k}") for k in range(6)):
                    top_depth = 1 + max(
                        (k for k in range(1, 6) if meta.get(f"layer_top_{k}")),
                        default=0,
                    )
                    tops, overlaps = fm.step_layer_top_overlap(meta, top_depth)
                    rec["step_layer_top20"] = tops
                    # Ids-only runs keep last-layer top-1 and penult top-3.
                    # Overlap, logit, prob, and cos θ are the bulky step fields.
                    ids_only = os.environ.get("OGF_LAYER_TOPK_IDS_ONLY", "0").strip().lower() not in (
                        "0",
                        "false",
                        "no",
                        "off",
                        "",
                    )
                    if not ids_only:
                        rec["step_layer_overlap"] = overlaps
                        if not table_dump and any(
                            meta.get(f"layer_top_logit_{k}") for k in range(6)
                        ):
                            rec["step_layer_top20_logit"] = fm.step_layer_top_vectors(
                                meta, "layer_top_logit_", top_depth
                            )
                        if any(meta.get(f"layer_top_prob_{k}") for k in range(6)):
                            rec["step_layer_top20_prob"] = fm.step_layer_top_vectors(
                                meta, "layer_top_prob_", top_depth
                            )
                        if any(meta.get(f"layer_top_cos_{k}") for k in range(6)):
                            rec["step_layer_top20_cos"] = fm.step_layer_top_vectors(
                                meta, "layer_top_cos_", top_depth
                            )
                        rec["mean_layer_overlap"] = fm.mean_overlap_columns(overlaps)
                        tail_w = fm.local_tail_window()
                        rec["mean_tail_layer_overlap"] = fm.mean_overlap_columns(
                            overlaps[-tail_w:]
                        )
            sample_deltas.append(rec)

        def _best_last_chunk(key):
            best_i = None
            best_s = None
            for j, rec in enumerate(sample_deltas):
                scores = rec.get(key) or []
                if not scores or scores[-1] is None:
                    continue
                score = float(scores[-1])
                if best_s is None or score > best_s:
                    best_s = score
                    best_i = j
            if best_i is None:
                return None
            return 1.0 if sample_deltas[best_i].get("correct") else 0.0

        best_chunk_c = _best_last_chunk("tree_chunk_c") if hold_keep else None
        best_chunk_dc = _best_last_chunk("tree_chunk_dc") if hold_keep else None
        div = _diversity_summary(token_id_lists)
        row = _case_record(dataset, sample)
        row.update(
            {
                "idx": start + k,
                "completion": case_texts if args.save_completions else None,
                "extracted": extracted,
                "generated_tokens": case_toks,
                "avg_generated_tokens": sum(case_toks) / max(n_kept, 1),
                "avg_think_tokens": sum(think_lens) / max(n_kept, 1),
                "avg_answer_tokens": sum(ans_lens) / max(n_kept, 1),
                "n": n_case,
                "n_kept": n_kept,
                "n_truncated": n_trunc,
                "kept_indices": keep,
                "finished": (
                    [bool(finished_flags[i]) for i in keep] if tree else [True] * n_kept
                ),
                "judge_info": judge_info,
                "passat1": sum(pass_list) / max(n_kept, 1),
                "vote_acc_best_chunk_c": best_chunk_c,
                "vote_acc_best_chunk_dc": best_chunk_dc,
                "passatk": max(pass_list) if pass_list else 0.0,
                "vote_acc": pass_list[first_index],
                "vote_acc_best_tail_c": vote_acc_best_tail_c,
                "vote_acc_best_tail_dc": selected["vote_acc_best_tail_dc"],
                "best_tail_dc_index": selected["best_tail_dc_index"],
                "vote_acc_best_dc": selected["vote_acc_best_dc"],
                "best_dc_index": selected["best_dc_index"],
                "vote_acc_majority": selected["vote_acc_majority"],
                "vote_acc_tail_dc_weighted": selected["vote_acc_tail_dc_weighted"],
                "tail_dc_weighted_index": selected["tail_dc_weighted_index"],
                "vote_acc_dc_weighted": selected["vote_acc_dc_weighted"],
                "dc_weighted_index": selected["dc_weighted_index"],
                "vote_key": voted,
                "vote_counts": {str(a): c for a, c in vote_counts.items()},
                "vote_acc_weighted": fm.weighted_vote_acc(pass_list, vote_keys, weights),
                "vote_acc_lo_ent": fm.vote_on_subset(pass_list, vote_keys, lo_ent_idx),
                "vote_acc_hi_logp": fm.vote_on_subset(pass_list, vote_keys, hi_logp_idx),
                "vote_acc_tail_c_weighted": paper["vote_acc_tail_c_weighted"],
                "vote_acc_tail_c_filter_30": paper["vote_acc_tail_c_filter_30"],
                "vote_acc_tail_c_filter_50": paper["vote_acc_tail_c_filter_50"],
                "passat1_tail_c_filter_30": paper["passat1_tail_c_filter_30"],
                "passat1_tail_c_filter_50": paper["passat1_tail_c_filter_50"],
                "vote_acc_joint_dc_weighted": paper["vote_acc_joint_dc_weighted"],
                "vote_acc_joint_dc_weighted_50": paper["vote_acc_joint_dc_weighted_50"],
                "vote_acc_joint_dc_weighted_60": paper["vote_acc_joint_dc_weighted_60"],
                "passat1_joint_50": paper["passat1_joint_50"],
                "passat1_joint_60": paper["passat1_joint_60"],
                "joint_retained": paper["joint_retained"],
                "joint_retained_50": paper["joint_retained_50"],
                "joint_retained_60": paper["joint_retained_60"],
                "joint_retained_indices": paper["joint_retained_indices"],
                "joint_rho": paper["joint_rho"],
                "tail_mode": tail_mode,
                "paper_tail_c_list": list(paper_c),
                "paper_tail_dc_list": list(paper_dc),
                "passat1_list": pass_list,
                "mean_logprob_list": logps,
                "mean_entropy_list": ents,
                "passat1_high_ent": p1_hi_ent,
                "passatk_high_ent": pk_hi_ent,
                "passat1_low_ent": p1_lo_ent,
                "passatk_low_ent": pk_lo_ent,
                "passat1_hi_logp": p1_hi_lp,
                "passatk_hi_logp": pk_hi_lp,
                "mean_last_mp": fm.mean_defined(last_mps),
                "mean_last_mcos": fm.mean_defined(last_mcs),
                "mean_last_c": fm.mean_defined(last_cs),
                "mean_all_dmp": fm.mean_defined(all_dmps),
                "mean_all_dmc": fm.mean_defined(all_dmcs),
                "mean_all_dc": fm.mean_defined(all_dcs),
                "mean_all_dc_n3": fm.mean_defined(all_dcs_n3),
                "mean_all_dc_n4": fm.mean_defined(all_dcs_n4),
                "mean_last_mp_list": last_mps,
                "mean_last_mcos_list": last_mcs,
                "mean_last_c_list": last_cs,
                "mean_all_dmp_list": all_dmps,
                "mean_all_dmc_list": all_dmcs,
                "mean_all_dc_list": all_dcs,
                "mean_all_dc_n3_list": all_dcs_n3,
                "mean_all_dc_n4_list": all_dcs_n4,
                "mean_match_dmp_list": match_dmps,
                "mean_match_dmc_list": match_dmcs,
                "mean_switch_dmp_list": switch_dmps,
                "mean_switch_dmc_list": switch_dmcs,
                "mean_tail_c_list": tail_cs,
                "mean_tail_dmp_list": tail_dmps,
                "mean_tail_dmc_list": tail_dmcs,
                "mean_tail_dc_list": tail_dcs,
                "mean_tail_dc_n3_list": tail_dcs_n3,
                "mean_tail_dc_n4_list": tail_dcs_n4,
                "mean_tail10_c_list": tail10_cs,
                "mean_tail10_dmp_list": tail10_dmps,
                "mean_tail10_dmc_list": tail10_dmcs,
                "mean_tail10_dc_list": tail10_dcs,
                "mean_tail10_dc_n3_list": tail10_dcs_n3,
                "mean_tail10_dc_n4_list": tail10_dcs_n4,
                "mean_tail20_c_list": tail20_cs,
                "mean_tail20_dmp_list": tail20_dmps,
                "mean_tail20_dmc_list": tail20_dmcs,
                "mean_tail20_dc_list": tail20_dcs,
                "mean_tail20_dc_n3_list": tail20_dcs_n3,
                "mean_tail20_dc_n4_list": tail20_dcs_n4,
                "passat1_hi_last_mp": p1_hi_lmp,
                "passatk_hi_last_mp": pk_hi_lmp,
                "passat1_hi_last_mcos": p1_hi_lmc,
                "passatk_hi_last_mcos": pk_hi_lmc,
                "passat1_hi_last_c": p1_hi_lc,
                "passatk_hi_last_c": pk_hi_lc,
                "passat1_hi_all_dmp": p1_hi_admp,
                "passatk_hi_all_dmp": pk_hi_admp,
                "passat1_hi_all_dmc": p1_hi_admc,
                "passatk_hi_all_dmc": pk_hi_admc,
                "passat1_hi_all_dc": p1_hi_adc,
                "passatk_hi_all_dc": pk_hi_adc,
                "passat1_hi_all_dc_n3": p1_hi_adc_n3,
                "passatk_hi_all_dc_n3": pk_hi_adc_n3,
                "passat1_hi_all_dc_n4": p1_hi_adc_n4,
                "passatk_hi_all_dc_n4": pk_hi_adc_n4,
                "passat1_hi_switch_dmp": p1_hi_sdmp,
                "passatk_hi_switch_dmp": pk_hi_sdmp,
                "passat1_hi_switch_dmc": p1_hi_sdmc,
                "passatk_hi_switch_dmc": pk_hi_sdmc,
                "passat1_hi_tail_c": p1_hi_tc,
                "passatk_hi_tail_c": pk_hi_tc,
                "passat1_hi_tail_dmp": p1_hi_tdmp,
                "passatk_hi_tail_dmp": pk_hi_tdmp,
                "passat1_hi_tail_dmc": p1_hi_tdmc,
                "passatk_hi_tail_dmc": pk_hi_tdmc,
                "passat1_hi_tail_dc": p1_hi_tdc,
                "passatk_hi_tail_dc": pk_hi_tdc,
                "passat1_hi_tail_dc_n3": p1_hi_tdc_n3,
                "passatk_hi_tail_dc_n3": pk_hi_tdc_n3,
                "passat1_hi_tail_dc_n4": p1_hi_tdc_n4,
                "passatk_hi_tail_dc_n4": pk_hi_tdc_n4,
                "passat1_hi_tail10_c": p1_hi_t10c,
                "passatk_hi_tail10_c": pk_hi_t10c,
                "passat1_hi_tail10_dmp": p1_hi_t10dmp,
                "passatk_hi_tail10_dmp": pk_hi_t10dmp,
                "passat1_hi_tail10_dmc": p1_hi_t10dmc,
                "passatk_hi_tail10_dmc": pk_hi_t10dmc,
                "passat1_hi_tail10_dc": p1_hi_t10dc,
                "passatk_hi_tail10_dc": pk_hi_t10dc,
                "passat1_hi_tail10_dc_n3": p1_hi_t10dc_n3,
                "passatk_hi_tail10_dc_n3": pk_hi_t10dc_n3,
                "passat1_hi_tail10_dc_n4": p1_hi_t10dc_n4,
                "passatk_hi_tail10_dc_n4": pk_hi_t10dc_n4,
                "passat1_hi_tail20_c": p1_hi_t20c,
                "passatk_hi_tail20_c": pk_hi_t20c,
                "passat1_hi_tail20_dmp": p1_hi_t20dmp,
                "passatk_hi_tail20_dmp": pk_hi_t20dmp,
                "passat1_hi_tail20_dmc": p1_hi_t20dmc,
                "passatk_hi_tail20_dmc": pk_hi_t20dmc,
                "passat1_hi_tail20_dc": p1_hi_t20dc,
                "passatk_hi_tail20_dc": pk_hi_t20dc,
                "passat1_hi_tail20_dc_n3": p1_hi_t20dc_n3,
                "passatk_hi_tail20_dc_n3": pk_hi_t20dc_n3,
                "passat1_hi_tail20_dc_n4": p1_hi_t20dc_n4,
                "passatk_hi_tail20_dc_n4": pk_hi_t20dc_n4,
                "sample_deltas": sample_deltas,
                "diversity": div,
            }
        )
        _append_case(row, k)

    judge_s = time.perf_counter() - t_judge0
    wall_s = time.perf_counter() - t_wall0
    scored = [r for r in results if int(r.get("n_kept") or 0) > 0]
    total = len(scored)
    n_tok = generated_token_total
    kept_toks = [
        int(x)
        for r in scored
        for x in (r.get("generated_tokens") or [])
        if x is not None
    ]
    n_drop = sum(int(r.get("n_truncated") or 0) for r in results)

    def avg(key):
        return sum(r[key] for r in scored) / total if total else 0.0

    def _tail_mean(score_key: str, correct: bool):
        xs = []
        for row in scored:
            passes = row.get("passat1_list") or []
            scores = row.get(score_key) or []
            for passed, score in zip(passes, scores):
                if score is None:
                    continue
                if (float(passed) >= 0.5) == correct:
                    xs.append(float(score))
        return fm.mean_defined(xs)

    stats = {
        "dataset": dataset,
        "data_path": data_path,
        "start_idx": start,
        "end_idx": end,
        "total_num": total,
        "n_truncated": n_drop,
        "n_kept": len(kept_toks),
        "num_samples": n,
        "filter_k": fk,
        "family": family,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "pass@1": avg("passat1"),
        "pass@k": avg("passatk"),
        "vote_acc": avg("vote_acc"),
        "vote_acc_best_tail_c": avg("vote_acc_best_tail_c"),
        "vote_acc_best_tail_dc": avg("vote_acc_best_tail_dc"),
        "keep_truncated": bool(getattr(args, "keep_truncated", False)),
        "vote_acc_best_dc": avg("vote_acc_best_dc"),
        "vote_acc_majority": avg("vote_acc_majority"),
        "vote_acc_tail_dc_weighted": avg("vote_acc_tail_dc_weighted"),
        "vote_acc_dc_weighted": avg("vote_acc_dc_weighted"),
        "vote_acc_tail_c_weighted": avg("vote_acc_tail_c_weighted"),
        "vote_acc_tail_c_filter_30": avg("vote_acc_tail_c_filter_30"),
        "pass@1_tail_c_filter_30": fm.avg_or_none(scored, "passat1_tail_c_filter_30"),
        "vote_acc_tail_c_filter_50": avg("vote_acc_tail_c_filter_50"),
        "pass@1_tail_c_filter_50": fm.avg_or_none(scored, "passat1_tail_c_filter_50"),
        "vote_acc_joint_dc_weighted_50": avg("vote_acc_joint_dc_weighted_50"),
        "vote_acc_joint_dc_weighted_60": avg("vote_acc_joint_dc_weighted_60"),
        "pass@1_joint_50": fm.avg_or_none(scored, "passat1_joint_50"),
        "pass@1_joint_60": fm.avg_or_none(scored, "passat1_joint_60"),
        "mean_joint_retained_50": avg("joint_retained_50"),
        "mean_joint_retained_60": avg("joint_retained_60"),
        "joint_rho": (
            float(args.joint_rho)
            if getattr(args, "joint_rho", None) is not None
            else agg.default_rho(family)
        ),
        "tail_mode": agg.resolve_tail_mode(family)[0],
        "vote_acc_weighted": avg("vote_acc_weighted"),
        "pass@1_high_ent": avg("passat1_high_ent"),
        "pass@k_high_ent": avg("passatk_high_ent"),
        "pass@1_low_ent": avg("passat1_low_ent"),
        "pass@k_low_ent": avg("passatk_low_ent"),
        "pass@1_hi_logp": avg("passat1_hi_logp"),
        "pass@k_hi_logp": avg("passatk_hi_logp"),
        "pass@1_hi_last_mp": fm.avg_or_none(scored, "passat1_hi_last_mp"),
        "pass@k_hi_last_mp": fm.avg_or_none(scored, "passatk_hi_last_mp"),
        "pass@1_hi_last_mcos": fm.avg_or_none(scored, "passat1_hi_last_mcos"),
        "pass@k_hi_last_mcos": fm.avg_or_none(scored, "passatk_hi_last_mcos"),
        "pass@1_hi_last_c": fm.avg_or_none(scored, "passat1_hi_last_c"),
        "pass@k_hi_last_c": fm.avg_or_none(scored, "passatk_hi_last_c"),
        "pass@1_hi_all_dmp": fm.avg_or_none(scored, "passat1_hi_all_dmp"),
        "pass@k_hi_all_dmp": fm.avg_or_none(scored, "passatk_hi_all_dmp"),
        "pass@1_hi_all_dmc": fm.avg_or_none(scored, "passat1_hi_all_dmc"),
        "pass@k_hi_all_dmc": fm.avg_or_none(scored, "passatk_hi_all_dmc"),
        "pass@1_hi_all_dc": fm.avg_or_none(scored, "passat1_hi_all_dc"),
        "pass@k_hi_all_dc": fm.avg_or_none(scored, "passatk_hi_all_dc"),
        "pass@1_hi_all_dc_n3": fm.avg_or_none(scored, "passat1_hi_all_dc_n3"),
        "pass@k_hi_all_dc_n3": fm.avg_or_none(scored, "passatk_hi_all_dc_n3"),
        "pass@1_hi_all_dc_n4": fm.avg_or_none(scored, "passat1_hi_all_dc_n4"),
        "pass@k_hi_all_dc_n4": fm.avg_or_none(scored, "passatk_hi_all_dc_n4"),
        "pass@1_hi_switch_dmp": fm.avg_or_none(scored, "passat1_hi_switch_dmp"),
        "pass@k_hi_switch_dmp": fm.avg_or_none(scored, "passatk_hi_switch_dmp"),
        "pass@1_hi_switch_dmc": fm.avg_or_none(scored, "passat1_hi_switch_dmc"),
        "pass@k_hi_switch_dmc": fm.avg_or_none(scored, "passatk_hi_switch_dmc"),
        "pass@1_hi_tail_c": fm.avg_or_none(scored, "passat1_hi_tail_c"),
        "pass@k_hi_tail_c": fm.avg_or_none(scored, "passatk_hi_tail_c"),
        "pass@1_hi_tail_dmp": fm.avg_or_none(scored, "passat1_hi_tail_dmp"),
        "pass@k_hi_tail_dmp": fm.avg_or_none(scored, "passatk_hi_tail_dmp"),
        "pass@1_hi_tail_dmc": fm.avg_or_none(scored, "passat1_hi_tail_dmc"),
        "pass@k_hi_tail_dmc": fm.avg_or_none(scored, "passatk_hi_tail_dmc"),
        "pass@1_hi_tail_dc": fm.avg_or_none(scored, "passat1_hi_tail_dc"),
        "pass@k_hi_tail_dc": fm.avg_or_none(scored, "passatk_hi_tail_dc"),
        "pass@1_hi_tail_dc_n3": fm.avg_or_none(scored, "passat1_hi_tail_dc_n3"),
        "pass@k_hi_tail_dc_n3": fm.avg_or_none(scored, "passatk_hi_tail_dc_n3"),
        "pass@1_hi_tail_dc_n4": fm.avg_or_none(scored, "passat1_hi_tail_dc_n4"),
        "pass@k_hi_tail_dc_n4": fm.avg_or_none(scored, "passatk_hi_tail_dc_n4"),
        "pass@1_hi_tail10_c": fm.avg_or_none(scored, "passat1_hi_tail10_c"),
        "pass@k_hi_tail10_c": fm.avg_or_none(scored, "passatk_hi_tail10_c"),
        "pass@1_hi_tail10_dmp": fm.avg_or_none(scored, "passat1_hi_tail10_dmp"),
        "pass@k_hi_tail10_dmp": fm.avg_or_none(scored, "passatk_hi_tail10_dmp"),
        "pass@1_hi_tail10_dmc": fm.avg_or_none(scored, "passat1_hi_tail10_dmc"),
        "pass@k_hi_tail10_dmc": fm.avg_or_none(scored, "passatk_hi_tail10_dmc"),
        "pass@1_hi_tail10_dc": fm.avg_or_none(scored, "passat1_hi_tail10_dc"),
        "pass@k_hi_tail10_dc": fm.avg_or_none(scored, "passatk_hi_tail10_dc"),
        "pass@1_hi_tail10_dc_n3": fm.avg_or_none(scored, "passat1_hi_tail10_dc_n3"),
        "pass@k_hi_tail10_dc_n3": fm.avg_or_none(scored, "passatk_hi_tail10_dc_n3"),
        "pass@1_hi_tail10_dc_n4": fm.avg_or_none(scored, "passat1_hi_tail10_dc_n4"),
        "pass@k_hi_tail10_dc_n4": fm.avg_or_none(scored, "passatk_hi_tail10_dc_n4"),
        "pass@1_hi_tail20_c": fm.avg_or_none(scored, "passat1_hi_tail20_c"),
        "pass@k_hi_tail20_c": fm.avg_or_none(scored, "passatk_hi_tail20_c"),
        "pass@1_hi_tail20_dmp": fm.avg_or_none(scored, "passat1_hi_tail20_dmp"),
        "pass@k_hi_tail20_dmp": fm.avg_or_none(scored, "passatk_hi_tail20_dmp"),
        "pass@1_hi_tail20_dmc": fm.avg_or_none(scored, "passat1_hi_tail20_dmc"),
        "pass@k_hi_tail20_dmc": fm.avg_or_none(scored, "passatk_hi_tail20_dmc"),
        "pass@1_hi_tail20_dc": fm.avg_or_none(scored, "passat1_hi_tail20_dc"),
        "pass@k_hi_tail20_dc": fm.avg_or_none(scored, "passatk_hi_tail20_dc"),
        "pass@1_hi_tail20_dc_n3": fm.avg_or_none(scored, "passat1_hi_tail20_dc_n3"),
        "pass@k_hi_tail20_dc_n3": fm.avg_or_none(scored, "passatk_hi_tail20_dc_n3"),
        "pass@1_hi_tail20_dc_n4": fm.avg_or_none(scored, "passat1_hi_tail20_dc_n4"),
        "pass@k_hi_tail20_dc_n4": fm.avg_or_none(scored, "passatk_hi_tail20_dc_n4"),
        "mean_last_mp": fm.mean_defined(r.get("mean_last_mp") for r in scored),
        "mean_last_mcos": fm.mean_defined(r.get("mean_last_mcos") for r in scored),
        "mean_last_c": fm.mean_defined(r.get("mean_last_c") for r in scored),
        "mean_all_dmp": fm.mean_defined(r.get("mean_all_dmp") for r in scored),
        "mean_all_dmc": fm.mean_defined(r.get("mean_all_dmc") for r in scored),
        "mean_all_dc": fm.mean_defined(r.get("mean_all_dc") for r in scored),
        "mean_all_dc_n3": fm.mean_defined(r.get("mean_all_dc_n3") for r in scored),
        "mean_all_dc_n4": fm.mean_defined(r.get("mean_all_dc_n4") for r in scored),
        "mean_tail_c_correct": _tail_mean("paper_tail_c_list", True),
        "mean_tail_c_incorrect": _tail_mean("paper_tail_c_list", False),
        "mean_tail_dc_correct": _tail_mean("paper_tail_dc_list", True),
        "mean_tail_dc_incorrect": _tail_mean("paper_tail_dc_list", False),
        "avg_token_length-all": (sum(kept_toks) / len(kept_toks)) if kept_toks else 0.0,
        "mean_unique_seq_ratio": (
            sum(r["diversity"].get("unique_ratio", 0.0) for r in scored) / total if total else 0.0
        ),
        "mean_logprob": fm.mean_defined(
            x for r in scored for x in (r.get("mean_logprob_list") or [])
        ),
        "mean_entropy": fm.mean_defined(
            x for r in scored for x in (r.get("mean_entropy_list") or [])
        ),
        "generate_s": generate_s,
        "judge_s": judge_s,
        "wall_s": wall_s,
        "completion_tokens": n_tok,
        "tokens_per_s": (n_tok / generate_s) if generate_s > 0 else 0.0,
    }
    if tree:
        stats["tree_search"] = True
        stats["tree_score"] = str(getattr(args, "tree_score", "dc") or "dc")
        stats["tree_keep"] = str(getattr(args, "tree_keep", "score") or "score")
        if stats["tree_keep"] not in ("", "score"):
            stats["pass@1_best_chunk_c"] = fm.avg_or_none(scored, "vote_acc_best_chunk_c")
            stats["pass@1_best_chunk_dc"] = fm.avg_or_none(scored, "vote_acc_best_chunk_dc")
        stats["tree_beam"] = int(args.tree_beam)
        stats["tree_branch"] = int(args.tree_branch)
        stats["tree_init_branch"] = int(getattr(args, "tree_init_branch", 0) or 0)
        stats["tree_chunk"] = int(args.tree_chunk)
        stats["tree_warmup"] = int(getattr(args, "tree_warmup", 0) or 0)
        stats["tree_rank"] = str(getattr(args, "tree_rank", "chunk") or "chunk")
        stats["tree_tail"] = int(getattr(args, "tree_tail", 2048) or 2048)
        stats["tree_rounds"] = max((info["rounds"] for info in (search_info or [])), default=0)
        stats["tree_expanded_tokens"] = n_tok
        stats["mean_final_traces"] = (
            sum(len(b) for b in bundles) / len(bundles) if bundles else 0.0
        )
    if bool(getattr(args, "keep_truncated", False)):
        trunc_note = f"truncated retained, over_length={n_drop}"
    else:
        trunc_note = f"truncated excluded, dropped={n_drop}"
    print(f"\n=== {dataset} statistics ({trunc_note}) ===")
    for key in _STAT_PRINT_KEYS:
        val = stats.get(key)
        if val is None:
            print(f"  {key}: n/a")
        elif isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    print(
        f"  time: generate={_fmt_hms(generate_s)}  judge={_fmt_hms(judge_s)}  "
        f"wall={_fmt_hms(wall_s)}  tokens={n_tok}  "
        f"{stats['tokens_per_s']:.1f} tok/s"
    )
    if tree:
        print(
            f"  tree: score={stats['tree_score']} keep={stats.get('tree_keep', 'score')} beam={stats['tree_beam']} "
            f"branch={stats['tree_branch']} init={stats['tree_init_branch']} "
            f"chunk={stats['tree_chunk']} warmup={stats['tree_warmup']} "
            f"rank={stats['tree_rank']} tail={stats['tree_tail']} "
            f"rounds={stats['tree_rounds']} "
            f"mean_finals={stats['mean_final_traces']:.2f} "
            f"expanded_tokens={stats['tree_expanded_tokens']}"
        )
    return {"stats": stats, "results": results}

