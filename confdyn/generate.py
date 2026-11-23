"""CLI entrypoint: over-generate then filter / majority-vote.

Core logic lives in sibling modules; this file keeps a stable import surface
for tests and ``python -m confdyn``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict

from confdyn import aggregation as agg
from confdyn.data import (
    COUNTDOWN_DATASETS,
    COUNTDOWN_TEMPLATE,
    COUNTDOWN_TEMPLATE_GEMMA4,
    DEFAULT_AIME,
    DEFAULT_COUNTDOWN6,
    DEFAULT_GPQA_DIAMOND,
    CODE_DATASETS,
    DEFAULT_HLE300,
    DEFAULT_HMMT,
    DEFAULT_LCB_V6,
    DEFAULT_MMMU_PRO150,
    DEFAULT_OLYMPIAD,
    DEFAULT_SGPQA200,
    EVAL_DATASETS,
    GPQA_QUERY_TEMPLATE,
    MATH_QA_DATASETS,
    MATH_QUERY_TEMPLATE,
    MCQ_DATASETS,
    VLM_MCQ_DATASETS,
    _REPO_ROOT,
    _case_record,
    _data_path,
    _dataset_defaults,
    _judge_sample,
    _load_jsonl,
    _load_samples,
    _math_answer,
    _math_question,
    _user_prompt,
)
from confdyn.dump_io import (
    _STEP_TRACE_KEYS,
    _as_output_list,
    _dump_step_stats_enabled,
    _fmt_hms,
    _write_step_sidecar,
)
from confdyn.engine import (
    _build_engine,
    _cuda_graph_max_bs,
    _expand_sample_major,
    _generate_chunk,
    _sampling_params,
    _sampling_params_batch,
    _sampling_params_for_traces,
    _trace_seed,
)
from confdyn.margins import (
    _approx_entropy,
    _diversity_summary,
    _log_margin_presence,
    _logprob_value,
    _mean_logprob,
    _meta_means,
    _top_logprob_values,
)
from confdyn.pipeline import _STAT_PRINT_KEYS, run_eval_dataset
from confdyn.prompting import (
    FAMILY_ENGINE,
    FAMILY_SAMPLING,
    GEMMA4_THINK_END,
    QWEN_THINK_END,
    _apply_chat,
    _apply_family_sampling_defaults,
    _strip_think,
    _think_answer_lens,
    _think_end_token,
    detect_family,
)
from confdyn.runtime_env import apply_default_env

apply_default_env()

# Re-exports used by tests / scripts (keep names stable).
__all__ = [
    "CODE_DATASETS",
    "COUNTDOWN_DATASETS",
    "COUNTDOWN_TEMPLATE",
    "COUNTDOWN_TEMPLATE_GEMMA4",
    "DEFAULT_AIME",
    "DEFAULT_COUNTDOWN6",
    "DEFAULT_GPQA_DIAMOND",
    "DEFAULT_HLE300",
    "DEFAULT_HMMT",
    "DEFAULT_LCB_V6",
    "DEFAULT_MMMU_PRO150",
    "DEFAULT_OLYMPIAD",
    "DEFAULT_SGPQA200",
    "EVAL_DATASETS",
    "FAMILY_ENGINE",
    "FAMILY_SAMPLING",
    "GEMMA4_THINK_END",
    "GPQA_QUERY_TEMPLATE",
    "MATH_QA_DATASETS",
    "MATH_QUERY_TEMPLATE",
    "MCQ_DATASETS",
    "QWEN_THINK_END",
    "VLM_MCQ_DATASETS",
    "_REPO_ROOT",
    "_STAT_PRINT_KEYS",
    "_STEP_TRACE_KEYS",
    "_apply_chat",
    "_apply_family_sampling_defaults",
    "_approx_entropy",
    "_as_output_list",
    "_build_engine",
    "_case_record",
    "_cuda_graph_max_bs",
    "_data_path",
    "_dataset_defaults",
    "_diversity_summary",
    "_dump_step_stats_enabled",
    "_expand_sample_major",
    "_fmt_hms",
    "_generate_chunk",
    "_judge_sample",
    "_load_jsonl",
    "_load_samples",
    "_log_margin_presence",
    "_logprob_value",
    "_math_answer",
    "_math_question",
    "_mean_logprob",
    "_meta_means",
    "_sampling_params",
    "_sampling_params_batch",
    "_sampling_params_for_traces",
    "_strip_think",
    "_think_answer_lens",
    "_think_end_token",
    "_top_logprob_values",
    "_trace_seed",
    "_user_prompt",
    "_write_step_sidecar",
    "detect_family",
    "main",
    "run_eval_dataset",
]


def main():
    parser = argparse.ArgumentParser(description="Over-generate then filter / majority-vote")
    parser.add_argument(
        "--model_path",
        default=os.path.join(_REPO_ROOT, "..", "models", "Qwen3.5-4B"),
    )
    parser.add_argument(
        "--family",
        default=None,
        choices=["qwen3", "qwen3.5", "qwen3.8", "gemma4", "nemotron3", "generic"],
    )
    parser.add_argument("--prompt", default="Briefly compute 17 * 23. Show steps.")
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--chat", action="store_true", help="Apply chat template (single-prompt mode)")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Sampling top_k. Default 20 for Qwen, 64 for Gemma 4, -1 (disabled) for Nemotron 3.",
    )
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=None,
        help="Default 1.5 for Qwen, 0.0 for Gemma 4 and Nemotron 3.",
    )
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="Data-parallel replicas. Each replica holds a full copy (TP=1). "
        "Gemma 4 12B fits one H100; use dp_size=2 on two GPUs.",
    )
    parser.add_argument("--mem_fraction_static", type=float, default=0.7)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--dump_json", default=None)
    parser.add_argument("--print_samples", action="store_true")
    parser.add_argument("--dataset", default=None, choices=list(EVAL_DATASETS))
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=100)
    parser.add_argument("--max_batch", type=int, default=1000000)
    parser.add_argument("--save_completions", action="store_true")
    parser.add_argument(
        "--keep_truncated",
        action="store_true",
        help="Keep length-capped traces in the pool. The default drops them before pass@1, voting, and tail scores.",
    )
    parser.add_argument(
        "--filter_k",
        type=int,
        default=5,
        help="Keep this many traces for legacy high/low-entropy and confidence filters.",
    )
    parser.add_argument(
        "--joint_rho",
        type=float,
        default=None,
        help="Retention ratio ρ for joint C@ρ ∩ ΔC@ρ (paper default: 0.6 Qwen, 0.5 Gemma).",
    )
    parser.add_argument("--return_logprob", action="store_true", default=True)
    parser.add_argument("--no_return_logprob", dest="return_logprob", action="store_false")
    parser.add_argument(
        "--top_logprobs_num",
        type=int,
        default=0,
        help="Top-k logprobs per step for approximate entropy. "
        "0 (default) uses -mean_logprob; 20 is slow (detokenize every step).",
    )
    parser.add_argument(
        "--cuda_graph_max_bs",
        type=int,
        default=None,
        help="Decode CUDA-graph max batch. Default: just above N * num_cases "
        "(H100 otherwise captures up to 256 and engine init takes minutes).",
    )
    parser.add_argument("--reasoning_parser", default=None)
    parser.add_argument("--attention_backend", default=None)
    parser.add_argument("--mamba_ssm_dtype", default=None, help="e.g. bfloat16 for Qwen3.8 GDN")
    parser.add_argument(
        "--linear_attn_backend",
        default=None,
        help="GDN kernel. Set triton to skip the FlashInfer auto-select.",
    )
    parser.add_argument("--linear_attn_decode_backend", default=None)
    parser.add_argument("--linear_attn_prefill_backend", default=None)
    parser.add_argument("--disable_cuda_graph", action="store_true")
    parser.add_argument(
        "--disable_custom_all_reduce",
        action="store_true",
        default=None,
        help="Use NCCL instead of sgl_kernel custom all-reduce (default on when tp>1).",
    )
    parser.add_argument(
        "--enable_custom_all_reduce",
        action="store_true",
        help="Force sgl_kernel custom all-reduce even for tp>1.",
    )
    parser.add_argument(
        "--tree_search",
        action="store_true",
        help="ΔC chunk tree search: expand `tree_branch` chunks, keep the "
        "`tree_beam` with the highest mean chunk ΔC, and repeat.",
    )
    parser.add_argument("--tree_beam", type=int, default=2, help="Live trajectories kept each round.")
    parser.add_argument(
        "--tree_branch",
        type=int,
        default=3,
        help="Chunks sampled from each live trajectory after the first round.",
    )
    parser.add_argument(
        "--tree_init_branch",
        type=int,
        default=0,
        help="Candidates sampled from the prompt in round 1. 0 uses tree_branch. 4 with tree_branch 2 is 4→2, then 2×2.",
    )
    parser.add_argument(
        "--tree_chunk",
        type=int,
        default=500,
        help="Fixed new-token length of each chunk after the warmup. EOS can end a chunk earlier.",
    )
    parser.add_argument(
        "--tree_warmup",
        type=int,
        default=0,
        help="First-round length. 0 uses tree_chunk. 3000 samples branch sequences of 3000 tokens before the 500-token chunks.",
    )
    parser.add_argument(
        "--tree_rank",
        default="chunk",
        choices=["chunk", "tail"],
        help="chunk: mean over the new tokens. tail: mean of the last tree_tail steps of the generation so far.",
    )
    parser.add_argument(
        "--tree_tail",
        type=int,
        default=2048,
        help="Window for tree_rank=tail. Qwen paper tail is 2048.",
    )
    parser.add_argument(
        "--tree_score",
        default="dc",
        choices=["dc", "c", "random"],
        help="Chunk ranking signal. dc: ΔC. c: final-layer C. "
        "random: draw beam chunks uniformly, ignoring the score.",
    )
    parser.add_argument(
        "--tree_keep",
        default="score",
        choices=["score", "c", "c4dc", "cguard", "c1dc", "c2dc"],
        help="score: rank by tree_score and keep every last-round chunk. "
        "c: keep the highest chunk-C children. "
        "c4dc: top 4 by chunk C, then the highest chunk ΔC. "
        "cguard: keep C rank 1, fill the other slots from C ranks 2-4 by ΔC. "
        "c1dc: keep the highest chunk C + chunk ΔC. "
        "c2dc: keep the highest chunk C + 2·chunk ΔC. "
        "These keep rules also hold a finished trajectory in its beam slot "
        "and drop unselected last-round chunks, so a problem ends with at most tree_beam traces.",
    )
    parser.add_argument(
        "--tree_last_all",
        action="store_true",
        help="On the last round, keep every candidate for voting. "
        "Earlier rounds still keep only tree_beam.",
    )
    args = parser.parse_args()
    if args.enable_custom_all_reduce:
        args.disable_custom_all_reduce = False
    elif args.disable_custom_all_reduce is None:
        args.disable_custom_all_reduce = args.tp_size > 1

    import sglang as sgl
    from transformers import AutoTokenizer

    model_path = os.path.abspath(args.model_path)
    family = detect_family(model_path, args.family)
    _apply_family_sampling_defaults(args, family)
    print(f"sglang from: {sgl.__file__}")
    print(f"model: {model_path}")
    print(
        f"family: {family}  filter_k={args.filter_k}  N={args.num_samples}  "
        f"joint_rho={args.joint_rho if args.joint_rho is not None else agg.default_rho(family)}  "
        f"tp={args.tp_size} dp={args.dp_size}  "
        f"T={args.temperature} top_p={args.top_p} top_k={args.top_k} "
        f"presence_penalty={args.presence_penalty}"
        + (
            f"  tree score={args.tree_score} keep={args.tree_keep} beam={args.tree_beam} "
            f"branch={args.tree_branch} init={args.tree_init_branch} chunk={args.tree_chunk}"
            if args.tree_search
            else ""
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    t_all0 = time.perf_counter()
    t_eng0 = time.perf_counter()
    llm = _build_engine(args, model_path, family)
    engine_s = time.perf_counter() - t_eng0
    print(f"engine ready in {_fmt_hms(engine_s)}", flush=True)
    payload: Dict[str, Any] = {}
    prompt = None
    try:
        if args.dataset in EVAL_DATASETS:
            payload = run_eval_dataset(args, tokenizer, llm, family)
        else:
            if args.prompt_file:
                with open(args.prompt_file, "r", encoding="utf-8") as f:
                    user = f.read().strip()
            else:
                user = args.prompt
            prompt = _apply_chat(tokenizer, user, family) if args.chat else user
            print(f"prompt ({len(prompt)} chars): {prompt[:200]!r}")
            prompts = [prompt] * args.num_samples
            sps = _sampling_params_batch(args, family, len(prompts), 0)
            outputs = _generate_chunk(llm, prompts, sps, args)
            texts = [o.get("text", "") for o in outputs]
            if args.print_samples:
                for i, t in enumerate(texts):
                    print(f"\n----- sample {i} -----\n{t}")
            payload = {
                "sampling_params": sps[0] if sps else _sampling_params(args, family),
                "texts": texts,
                "n": len(texts),
            }
    finally:
        llm.shutdown()

    total_s = time.perf_counter() - t_all0
    if isinstance(payload, dict) and isinstance(payload.get("stats"), dict):
        payload["stats"]["engine_s"] = engine_s
        payload["stats"]["total_s"] = total_s
    print(
        f"total running time: {_fmt_hms(total_s)}  "
        f"(engine={_fmt_hms(engine_s)})",
        flush=True,
    )

    if args.dump_json:
        out = os.path.abspath(args.dump_json)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        step_path = _write_step_sidecar(out, payload)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {"args": vars(args), "prompt": prompt, "runs": payload},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"wrote {out}")
        if step_path:
            print(f"wrote step traces {step_path}")
        elif _dump_step_stats_enabled():
            print(
                "[confdyn] WARNING: DUMP_STEP_STATS=1 but no .steps.jsonl "
                "(last_c / penult_dc lists were empty)",
                flush=True,
            )


if __name__ == "__main__":
    main()
