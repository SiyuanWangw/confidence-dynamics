#!/usr/bin/env python3
"""Sample N reasoning trajectories, then aggregate with confidence dynamics.

Implements the paper method:
  joint filter C@ρ ∩ ΔC@ρ (tail) → ΔC-weighted majority vote,
plus baselines (unweighted majority, tail-C weighted, tail-C filter @20/30/50%).

Requires the local SGLang tree in ``sglang_pkg/`` (layer-margin hooks for C / ΔC).

Example:
  python generate.py \\
    --model_path /path/to/Qwen3-8B \\
    --dataset hmmt --num_samples 10 --joint_rho 0.6 \\
    --dump_json results/hmmt_smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple


def _pin_torch_nccl_libs() -> None:
    """Prefer system NCCL over a mismatched pip nvidia-nccl when present."""
    distro = "/usr/lib/x86_64-linux-gnu"
    if os.path.isdir(distro):
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in cur.split(":") if p]
        if distro not in parts:
            os.environ["LD_LIBRARY_PATH"] = ":".join([distro, *parts]) if parts else distro
        elif parts and parts[0] != distro:
            parts = [p for p in parts if p != distro]
            os.environ["LD_LIBRARY_PATH"] = ":".join([distro, *parts])
    os.environ.setdefault("NCCL_NET_PLUGIN", "none")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")


_pin_torch_nccl_libs()
os.environ.setdefault("OGF_LAYER_MARGINS", "1")
os.environ.setdefault("DUMP_STEP_STATS", "1")
os.environ.setdefault("LOCAL_TAIL_WINDOW", "2048")
os.environ.setdefault("DEEPCONF_K", "20")
# Scheduler is a spawned subprocess: it inherits env, not the parent logger.
os.environ.setdefault("SGLANG_LOG_LEVEL", "error")

import aggregation as agg
import countdown_eval
import filter_metrics as fm
import gpqa_eval
import gsm8k_eval
import hle_eval
import hmmt_eval
import olympiad_eval

COUNTDOWN_TEMPLATE = """
{Question}
You may use basic arithmetic operations (+, -, *, /), and each number can only be used once. Please reason step by step and show your reasoning inside <think> </think> tags. Then provide only the final equation enclosed in \\boxed{{}}, for example: \\boxed{{(1 + 2) / 3}}.
"""
COUNTDOWN_TEMPLATE_GEMMA4 = """
{Question}
You may use basic arithmetic operations (+, -, *, /), and each number can only be used once. Please reason step by step. Then provide only the final equation enclosed in \\boxed{{}}, for example: \\boxed{{(1 + 2) / 3}}.
"""

MATH_QUERY_TEMPLATE = (
    "{Question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
)
GPQA_QUERY_TEMPLATE = gpqa_eval.query_template()

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DATA = os.path.join(_REPO_ROOT, "datasets")
# Optional fallback to a sibling Soft-Thinking checkout (dev convenience).
_SOFT_THINKING_DATA = os.path.join(_REPO_ROOT, "..", "Soft-Thinking", "datasets")


def _data_path(*parts: str) -> str:
    local = os.path.join(_LOCAL_DATA, *parts)
    if os.path.exists(local):
        return local
    return os.path.join(_SOFT_THINKING_DATA, *parts)


DEFAULT_COUNTDOWN5 = _data_path("countdown5.jsonl")
DEFAULT_COUNTDOWN6 = _data_path("countdown6.jsonl")
DEFAULT_OLYMPIAD = _data_path("olympiad", "test.jsonl")
DEFAULT_GSM8K_P2 = _data_path("gsm8k_p2.jsonl")
DEFAULT_AIME = _data_path("aime.json")
DEFAULT_AIME26 = _data_path("aime2026.json")
DEFAULT_HMMT = _data_path("hmmt.json")
DEFAULT_GPQA_DIAMOND = _data_path("gpqa_diamond.json")
DEFAULT_HLE300 = _data_path("hle300.json")
DEFAULT_SGPQA200 = _data_path("sgpqa200.json")
MATH_QA_DATASETS = ("olympiad", "gsm8k_p2", "aime", "aime26", "hmmt", "hle300")
MCQ_DATASETS = ("gpqa_diamond", "sgpqa200")
COUNTDOWN_DATASETS = ("countdown5", "countdown6")
EVAL_DATASETS = COUNTDOWN_DATASETS + MATH_QA_DATASETS + MCQ_DATASETS

FAMILY_ENGINE = {
    "qwen3.8": {"reasoning_parser": "qwen3"},
    "qwen3.5": {"reasoning_parser": "qwen3"},
    "qwen3": {"reasoning_parser": "qwen3"},
    "gemma4": {"reasoning_parser": "gemma4"},
}
# Gemma 4 docs: T=1.0, top_p=0.95, top_k=64. Qwen thinking runs keep presence_penalty=1.5.
FAMILY_SAMPLING = {
    "gemma4": {"top_k": 64, "presence_penalty": 0.0, "skip_special_tokens": False},
}
GEMMA4_THINK_END = "<channel|>"
QWEN_THINK_END = "</think>"

_STAT_PRINT_KEYS = (
    "total_num",
    "pass@1",
    "pass@k",
    "vote_acc",
    "vote_acc_tail_c_weighted",
    "vote_acc_tail_c_filter_20",
    "vote_acc_tail_c_filter_30",
    "vote_acc_tail_c_filter_50",
    "vote_acc_joint_dc_weighted",
    "mean_joint_retained",
    "joint_rho",
    "tail_mode",
    "vote_acc_weighted",
    "vote_acc_lo_ent",
    "vote_acc_hi_logp",
    "pass@1_hi_tail_c",
    "pass@1_hi_tail_dc",
    "pass@1_hi_tail20_c",
    "pass@1_hi_tail20_dc",
    "mean_last_c",
    "mean_all_dc",
    "mean_logprob",
    "mean_entropy",
    "avg_token_length-all",
    "mean_unique_seq_ratio",
    "generate_s",
    "judge_s",
    "wall_s",
)


def _fmt_hms(seconds: float) -> str:
    s = max(float(seconds), 0.0)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - 3600 * h - 60 * m
    if h:
        return f"{h}h{m:02d}m{sec:04.1f}s"
    if m:
        return f"{m}m{sec:04.1f}s"
    return f"{sec:.1f}s"


_STEP_TRACE_KEYS = (
    "step_c",
    "step_dc",
    "step_dc_n3",
    "step_dc_n4",
    "step_match",
)


def _dump_step_stats_enabled() -> bool:
    return os.environ.get("DUMP_STEP_STATS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _write_step_sidecar(dump_path: str, payload: Any) -> Optional[str]:
    """Move per-decode-step lists out of the main dump, like angular-sampling."""
    if not isinstance(payload, dict):
        return None
    traces = []
    for row in payload.get("results") or []:
        idx = row.get("idx")
        for sd in row.get("sample_deltas") or []:
            rec = {"idx": idx, "sample": sd.get("sample")}
            n = 0
            for key in _STEP_TRACE_KEYS:
                xs = sd.pop(key, None)
                if xs:
                    rec[key] = xs
                    n = max(n, len(xs))
            if n:
                rec["steps"] = n
                traces.append(rec)
    if not traces:
        return None
    step_path = (
        dump_path[:-5] + ".steps.jsonl"
        if dump_path.endswith(".json")
        else dump_path + ".steps.jsonl"
    )
    with open(step_path, "w", encoding="utf-8") as f:
        for rec in traces:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
    return step_path


def _as_output_list(outputs) -> List[Dict[str, Any]]:
    if outputs is None:
        return []
    if isinstance(outputs, dict):
        return [outputs]
    return list(outputs)


def _load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_samples(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        return _load_jsonl(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"expected a JSON list in {path}")


def _think_end_token(family: str) -> str:
    if family == "gemma4":
        return GEMMA4_THINK_END
    return QWEN_THINK_END


def _strip_think(text: str, family: str = "") -> str:
    token = _think_end_token(family)
    if token in text:
        return text.rpartition(token)[2]
    for alt in (QWEN_THINK_END, GEMMA4_THINK_END):
        if alt != token and alt in text:
            return text.rpartition(alt)[2]
    return text


def _think_answer_lens(text: str, tokenizer, think_end: str = QWEN_THINK_END) -> Tuple[int, int]:
    if think_end in text:
        think, _, ans = text.partition(think_end)
        return len(tokenizer.encode(think)), len(tokenizer.encode(ans))
    return len(tokenizer.encode(text)), 0


def detect_family(model_path: str, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    blob = os.path.basename(os.path.abspath(model_path)).lower().replace("_", ".")
    if "gemma-4" in blob or "gemma4" in blob:
        return "gemma4"
    if "qwen3.8" in blob or "qwen3-8" in blob:
        return "qwen3.8"
    if "qwen3.5" in blob:
        return "qwen3.5"
    if "qwen3" in blob:
        return "qwen3"
    return "generic"


def _apply_family_sampling_defaults(args, family: str) -> None:
    spec = FAMILY_SAMPLING.get(family, {})
    if getattr(args, "top_k", None) is None:
        args.top_k = int(spec.get("top_k", 20))
    if getattr(args, "presence_penalty", None) is None:
        args.presence_penalty = float(spec.get("presence_penalty", 1.5))


def _apply_chat(tokenizer, user_content: str, family: str) -> str:
    user = [{"role": "user", "content": user_content}]
    kw = dict(add_generation_prompt=True, tokenize=False)
    if family.startswith("qwen") or family == "gemma4":
        try:
            return tokenizer.apply_chat_template(user, enable_thinking=True, **kw)
        except TypeError:
            pass
    if family == "gemma4":
        # Docs: thinking is on when <|think|> starts the system prompt.
        chat = [
            {"role": "system", "content": "<|think|>"},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(chat, **kw)
    return tokenizer.apply_chat_template(user, **kw)


def _dataset_defaults(name: str) -> str:
    if name == "olympiad":
        return DEFAULT_OLYMPIAD
    if name == "gsm8k_p2":
        return DEFAULT_GSM8K_P2
    if name == "aime":
        return DEFAULT_AIME
    if name == "aime26":
        return DEFAULT_AIME26
    if name == "hmmt":
        return DEFAULT_HMMT
    if name == "gpqa_diamond":
        return DEFAULT_GPQA_DIAMOND
    if name == "sgpqa200":
        return DEFAULT_SGPQA200
    if name == "hle300":
        return DEFAULT_HLE300
    if name == "countdown5":
        return DEFAULT_COUNTDOWN5
    return DEFAULT_COUNTDOWN6


def _math_question(sample: dict) -> str:
    if sample.get("question"):
        return sample["question"]
    prompt = sample.get("prompt")
    if isinstance(prompt, list) and prompt:
        return prompt[0]["value"]
    if isinstance(prompt, str):
        return prompt
    raise KeyError("sample is missing question/prompt")


def _math_answer(sample: dict) -> str:
    if sample.get("answer") is not None:
        return str(sample["answer"])
    return str(sample["final_answer"])


def _user_prompt(dataset: str, sample: dict, family: str = "") -> str:
    if dataset in MCQ_DATASETS:
        letters = gpqa_eval.choice_letters(sample)
        return gpqa_eval.query_template(letters).format(
            Question=gpqa_eval.format_gpqa_question(sample)
        )
    if dataset in MATH_QA_DATASETS:
        return MATH_QUERY_TEMPLATE.format(Question=_math_question(sample))
    countdown = COUNTDOWN_TEMPLATE_GEMMA4 if family == "gemma4" else COUNTDOWN_TEMPLATE
    return countdown.format(Question=sample["prompt"])


def _judge_sample(
    dataset: str, text: str, sample: dict, finished: bool, family: str = ""
) -> Tuple[bool, str]:
    text = _strip_think(text, family)
    if dataset == "olympiad":
        return olympiad_eval.rule_judge(text, _math_answer(sample), finished)
    if dataset == "hmmt":
        return hmmt_eval.rule_judge(text, _math_answer(sample), finished)
    if dataset in ("gsm8k_p2", "aime", "aime26"):
        return gsm8k_eval.rule_judge(text, _math_answer(sample), finished)
    if dataset == "hle300":
        return hle_eval.rule_judge(text, sample, finished)
    if dataset in MCQ_DATASETS:
        return gpqa_eval.rule_judge(
            text,
            sample["final_answer"],
            finished,
            letters=gpqa_eval.choice_letters(sample),
        )
    if dataset in COUNTDOWN_DATASETS:
        return countdown_eval.rule_judge(
            text, int(sample["target"]), list(sample["nums"]), variant=dataset
        )
    raise ValueError(f"unknown dataset {dataset}")


def _case_record(dataset: str, sample: dict) -> dict:
    if dataset in MCQ_DATASETS:
        rec = {
            "prompt": gpqa_eval.format_gpqa_question(sample),
            "ground_truth": sample["final_answer"],
            "choices": sample.get("choices"),
        }
        if sample.get("id"):
            rec["id"] = sample["id"]
        if sample.get("field"):
            rec["field"] = sample["field"]
        return rec
    if dataset in MATH_QA_DATASETS:
        rec = {
            "prompt": _math_question(sample),
            "ground_truth": _math_answer(sample),
            "id": sample.get("id"),
        }
        if sample.get("source"):
            rec["source"] = sample["source"]
        if sample.get("kind"):
            rec["kind"] = sample["kind"]
        return rec
    return {
        "prompt": sample["prompt"],
        "ground_truth": sample["target"],
        "nums": sample["nums"],
        "answer": sample.get("answer"),
    }


def _logprob_value(item) -> Optional[float]:
    if item is None:
        return None
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, (list, tuple)) and item:
        try:
            return float(item[0])
        except (TypeError, ValueError):
            return None
    if isinstance(item, dict):
        for key in ("logprob", "log_prob", "lp"):
            if item.get(key) is not None:
                return float(item[key])
    return None


def _mean_logprob(meta: dict) -> Optional[float]:
    toks = meta.get("output_token_logprobs") or []
    xs = [v for v in (_logprob_value(x) for x in toks) if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _top_logprob_values(step) -> List[float]:
    if not step:
        return []
    if isinstance(step, dict):
        out = []
        for v in step.values():
            lp = _logprob_value(v) if not isinstance(v, (int, float)) else float(v)
            if lp is not None:
                out.append(lp)
        return out
    if isinstance(step, (list, tuple)):
        out = []
        for item in step:
            lp = _logprob_value(item)
            if lp is not None:
                out.append(lp)
        return out
    return []


def _approx_entropy(meta: dict) -> Optional[float]:
    """Shannon entropy of the returned top-k, renormalized (upper-bounded by log k)."""
    steps = meta.get("output_top_logprobs") or []
    ents = []
    for step in steps:
        lps = _top_logprob_values(step)
        if len(lps) < 2:
            continue
        m = max(lps)
        exps = [math.exp(x - m) for x in lps]
        z = sum(exps)
        if z <= 0:
            continue
        ps = [e / z for e in exps]
        ents.append(-sum(p * math.log(max(p, 1e-12)) for p in ps))
    if ents:
        return sum(ents) / len(ents)
    # Fallback: higher (-mean logprob) ≈ more uncertain.
    mlp = _mean_logprob(meta)
    if mlp is None:
        return None
    return max(-float(mlp), 0.0)


def _log_margin_presence(metas: List[dict], family: str) -> None:
    """Fail-loud: C / ΔC lists must land in meta_info or dump averages stay None."""
    n = len(metas)
    n_c = sum(1 for m in metas if m.get("last_c"))
    n_dc = sum(1 for m in metas if m.get("penult_dc"))
    n_dc3 = sum(1 for m in metas if m.get("n3_dc"))
    n_dc4 = sum(1 for m in metas if m.get("n4_dc"))
    n_c_steps = max((len(m.get("last_c") or []) for m in metas), default=0)
    n_dc_steps = max((len(m.get("penult_dc") or []) for m in metas), default=0)
    flags = (
        f"family={family} OGF_LAYER_MARGINS={os.environ.get('OGF_LAYER_MARGINS', '1')} "
        f"DUMP_STEP_STATS={os.environ.get('DUMP_STEP_STATS', '1')} "
        f"DEEPCONF_K={os.environ.get('DEEPCONF_K', '20')}"
    )
    if n_c == 0 or n_dc == 0:
        print(
            f"[ogf] ERROR: C/ΔC missing after first batch ({n} seqs, "
            f"last_c={n_c}/{n} penult_dc={n_dc}/{n} n3_dc={n_dc3}/{n} n4_dc={n_dc4}/{n}; "
            f"{flags}). mean_last_c_list / mean_all_dc_list will be null.",
            flush=True,
        )
        return
    print(
        f"[ogf] C/ΔC present: last_c={n_c}/{n} (max {n_c_steps} steps) "
        f"penult_dc={n_dc}/{n} (max {n_dc_steps} steps) "
        f"n3_dc={n_dc3}/{n} n4_dc={n_dc4}/{n}; {flags}",
        flush=True,
    )


def _meta_means(meta: dict, tail_window: Optional[int] = None) -> dict:
    last_mp = fm.mean_steps(meta.get("last_mp"))
    last_mcos = fm.mean_steps(meta.get("last_mcos"))
    last_c = meta.get("last_c")
    dmp = meta.get("penult_dmp")
    dmc = meta.get("penult_dmc")
    dc = meta.get("penult_dc")
    dc_n3 = meta.get("n3_dc")
    dc_n4 = meta.get("n4_dc")
    match = meta.get("penult_match")
    w = fm.local_tail_window() if tail_window is None else max(int(tail_window), 1)
    return {
        "mean_last_mp": last_mp,
        "mean_last_mcos": last_mcos,
        "mean_last_c": fm.mean_steps(last_c),
        "mean_all_dmp": fm.mean_steps(dmp),
        "mean_all_dmc": fm.mean_steps(dmc),
        "mean_all_dc": fm.mean_steps(dc),
        "mean_all_dc_n3": fm.mean_steps(dc_n3),
        "mean_all_dc_n4": fm.mean_steps(dc_n4),
        "mean_match_dmp": fm.mean_steps_where(dmp, match),
        "mean_match_dmc": fm.mean_steps_where(dmc, match),
        "mean_match_dc": fm.mean_steps_where(dc, match),
        "mean_switch_dmp": fm.mean_steps_where(dmp, match, invert=True),
        "mean_switch_dmc": fm.mean_steps_where(dmc, match, invert=True),
        "mean_switch_dc": fm.mean_steps_where(dc, match, invert=True),
        "mean_tail_c": fm.mean_tail_steps(last_c, w),
        "mean_tail_dmp": fm.mean_tail_steps(dmp, w),
        "mean_tail_dmc": fm.mean_tail_steps(dmc, w),
        "mean_tail_dc": fm.mean_tail_steps(dc, w),
        "mean_tail_dc_n3": fm.mean_tail_steps(dc_n3, w),
        "mean_tail_dc_n4": fm.mean_tail_steps(dc_n4, w),
        "mean_tail10_c": fm.mean_tail_frac(last_c, 0.10),
        "mean_tail10_dmp": fm.mean_tail_frac(dmp, 0.10),
        "mean_tail10_dmc": fm.mean_tail_frac(dmc, 0.10),
        "mean_tail10_dc": fm.mean_tail_frac(dc, 0.10),
        "mean_tail10_dc_n3": fm.mean_tail_frac(dc_n3, 0.10),
        "mean_tail10_dc_n4": fm.mean_tail_frac(dc_n4, 0.10),
        "mean_tail20_c": fm.mean_tail_frac(last_c, 0.20),
        "mean_tail20_dmp": fm.mean_tail_frac(dmp, 0.20),
        "mean_tail20_dmc": fm.mean_tail_frac(dmc, 0.20),
        "mean_tail20_dc": fm.mean_tail_frac(dc, 0.20),
        "mean_tail20_dc_n3": fm.mean_tail_frac(dc_n3, 0.20),
        "mean_tail20_dc_n4": fm.mean_tail_frac(dc_n4, 0.20),
    }


def _diversity_summary(token_ids: List[List[int]]) -> dict:
    n = len(token_ids)
    uniq = len({tuple(x) for x in token_ids})
    lens = [len(x) for x in token_ids]
    unigrams = [tok for seq in token_ids for tok in seq]
    bigrams = [pair for seq in token_ids for pair in zip(seq, seq[1:])]
    d1 = (len(set(unigrams)) / len(unigrams)) if unigrams else 0.0
    d2 = (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    return {
        "n": n,
        "unique_seq": uniq,
        "unique_ratio": uniq / max(n, 1),
        "distinct_1": d1,
        "distinct_2": d2,
        "mean_len": (sum(lens) / n) if n else 0.0,
    }


def _cuda_graph_max_bs(args) -> int:
    """H100 default captures decode graphs up to bs=256 (~36 shapes).

    Qwen3.5 GDN capture is several seconds per shape, so a 2-prompt N=10
    smoke spent ~2.5 min in engine init. Capture just above the expected
    concurrent batch instead.
    """
    if getattr(args, "cuda_graph_max_bs", None) is not None:
        return max(int(args.cuda_graph_max_bs), 1)
    n_cases = 1
    if args.dataset:
        n_cases = max(int(args.end_idx) - int(args.start_idx), 1)
    concurrent = max(int(args.num_samples), 1) * n_cases
    dp = max(int(getattr(args, "dp_size", 1) or 1), 1)
    concurrent = (concurrent + dp - 1) // dp
    if concurrent <= 32:
        return 32
    if concurrent <= 64:
        return 64
    if concurrent <= 160:
        return 160
    return 256


def _build_engine(args, model_path: str, family: str):
    import sglang as sgl

    log_level = os.environ.get("SGLANG_LOG_LEVEL", "error")
    kwargs = dict(
        model_path=model_path,
        tp_size=args.tp_size,
        dp_size=max(int(getattr(args, "dp_size", 1) or 1), 1),
        log_level=log_level,
        trust_remote_code=True,
        random_seed=args.random_seed,
        mem_fraction_static=args.mem_fraction_static,
        cuda_graph_max_bs_decode=_cuda_graph_max_bs(args),
    )
    # Decode/Prefill batch lines are logger.info; HF load can re-enable INFO.
    # Skip the periodic dump unless the caller asked for info/debug.
    if log_level.lower() not in ("debug", "info"):
        kwargs["decode_log_interval"] = 10**9
    kwargs.update(FAMILY_ENGINE.get(family, {}))
    if args.reasoning_parser:
        kwargs["reasoning_parser"] = args.reasoning_parser
    if args.disable_cuda_graph:
        kwargs["disable_cuda_graph"] = True
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if args.mamba_ssm_dtype:
        kwargs["mamba_ssm_dtype"] = args.mamba_ssm_dtype
    # TP>1 uses sgl_kernel custom all-reduce by default. ogf has sgl_kernel
    # 0.4.6 + torch cu129 / driver 550; that path often SIGSEGV while TP=1
    # (no all-reduce) works. NCCL is the stable fallback.
    if args.disable_custom_all_reduce:
        kwargs["disable_custom_all_reduce"] = True
    print(f"Engine kwargs: {kwargs}", flush=True)
    return sgl.Engine(**kwargs)


def _trace_seed(args, case_idx: int, sample_idx: int, n_samples: int) -> int:
    """Unique seed per (question, sample). Independent of dp_size (2 or 4).

    DP round-robin would otherwise put N clones of the same prompt on
    different ranks at the same local index and replay Engine.random_seed.
    """
    base = int(getattr(args, "random_seed", 0) or 0)
    return base + int(case_idx) * int(n_samples) + int(sample_idx)


def _expand_sample_major(
    case_prompts: List[str], n_samples: int
) -> Tuple[List[str], List[int]]:
    """s0 for all cases, then s1, ... so DP ranks do not align clones."""
    nq = len(case_prompts)
    flat, dest = [], []
    for sample_idx in range(n_samples):
        for case_local in range(nq):
            flat.append(case_prompts[case_local])
            dest.append(case_local * n_samples + sample_idx)
    return flat, dest


def _sampling_params(
    args, family: str = "", sampling_seed: Optional[int] = None
) -> dict:
    sp = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "n": 1,
        "max_new_tokens": args.max_new_tokens,
    }
    spec = FAMILY_SAMPLING.get(family, {})
    if "skip_special_tokens" in spec:
        sp["skip_special_tokens"] = spec["skip_special_tokens"]
    if sampling_seed is not None:
        sp["sampling_seed"] = int(sampling_seed)
    return sp


def _sampling_params_for_traces(
    args, family: str, dest: List[int], n_samples: int, case0: int
) -> List[dict]:
    """One sampling dict per request; seed = random_seed + case * N + sample."""
    sps = []
    for dest_idx in dest:
        case_local, sample_idx = divmod(int(dest_idx), int(n_samples))
        sps.append(
            _sampling_params(
                args,
                family,
                sampling_seed=_trace_seed(args, case0 + case_local, sample_idx, n_samples),
            )
        )
    return sps


def _sampling_params_batch(
    args, family: str, n: int, start: int, n_samples: Optional[int] = None
) -> List[dict]:
    """Seeds for a single-prompt N-copy batch (sample_idx = 0..n-1)."""
    ns = int(n_samples or n)
    case0 = int(getattr(args, "start_idx", 0) or 0)
    return [
        _sampling_params(args, family, sampling_seed=_trace_seed(args, case0, start + i, ns))
        for i in range(n)
    ]


def _generate_chunk(
    llm,
    prompts: List[str],
    sp: Any,
    args,
) -> List[dict]:
    kw = dict(return_logprob=bool(args.return_logprob))
    if args.return_logprob and args.top_logprobs_num > 0:
        kw["top_logprobs_num"] = args.top_logprobs_num
    try:
        outputs = llm.generate(prompts, sp, **kw)
    except TypeError:
        outputs = llm.generate(prompts, sp)
    return _as_output_list(outputs)


def run_eval_dataset(args, tokenizer, llm, family: str) -> dict:
    dataset = args.dataset
    data_path = os.path.abspath(args.data_path or _dataset_defaults(dataset))
    samples = _load_samples(data_path)
    start = args.start_idx
    end = min(args.end_idx, len(samples))
    subset = samples[start:end]
    n = args.num_samples
    fk = max(int(args.filter_k), 1)
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

    case_prompts = [
        _apply_chat(tokenizer, _user_prompt(dataset, sample, family), family)
        for sample in subset
    ]
    # Sample-major so DP=2/4 round-robin does not pair clones of the same prompt.
    prompts, dest = _expand_sample_major(case_prompts, n)
    seed0 = _trace_seed(args, start, 0, n)
    seed1 = _trace_seed(args, start + max(len(subset) - 1, 0), max(n - 1, 0), n)
    print(
        f"Generating {len(prompts)} sequences: {_sampling_params(args, family)} "
        f"dp={max(int(getattr(args, 'dp_size', 1) or 1), 1)} "
        f"layout=sample-major sampling_seed={seed0}..{seed1}",
        flush=True,
    )

    n_out = len(prompts)
    texts: List[Optional[str]] = [None] * n_out
    token_counts: List[Optional[int]] = [None] * n_out
    finished: List[Optional[bool]] = [None] * n_out
    metas: List[Optional[dict]] = [None] * n_out
    batch = max(int(args.max_batch), 1)
    i = 0
    t_wall0 = time.perf_counter()
    t_gen0 = time.perf_counter()
    first_chunk = True
    while i < len(prompts):
        chunk = prompts[i : i + batch]
        chunk_dest = dest[i : i + batch]
        print(f"  batch {i}:{i + len(chunk)} / {len(prompts)}", flush=True)
        sps = _sampling_params_for_traces(args, family, chunk_dest, n, start)
        outs = _generate_chunk(llm, chunk, sps, args)
        if len(outs) != len(chunk):
            raise RuntimeError(
                f"generate returned {len(outs)} outputs for {len(chunk)} prompts"
            )
        for o, d in zip(outs, chunk_dest):
            texts[d] = o.get("text", "")
            meta = o.get("meta_info") or {}
            metas[d] = meta
            token_counts[d] = int(meta.get("completion_tokens") or 0)
            reason = meta.get("finish_reason") or {}
            if isinstance(reason, dict):
                finished[d] = reason.get("type") == "stop"
            else:
                finished[d] = str(reason) in ("stop", "eos", "True")
        i += len(chunk)
        if first_chunk:
            _log_margin_presence([m for m in metas if m is not None], family)
            first_chunk = False
    if any(t is None for t in texts):
        raise RuntimeError("sample-major scatter missed some traces")
    generate_s = time.perf_counter() - t_gen0
    t_judge0 = time.perf_counter()

    results = []
    for k, sample in enumerate(subset):
        sl = slice(k * n, (k + 1) * n)
        case_texts = texts[sl]
        case_toks = token_counts[sl]
        case_fin = finished[sl]
        case_meta = metas[sl]
        pass_list = []
        extracted = []
        judge_info = []
        token_id_lists = []
        for text, fin in zip(case_texts, case_fin):
            try:
                ok, extracted_answer = _judge_sample(dataset, text, sample, fin, family)
            except Exception as exc:
                print(f"judge error idx={start + k}: {exc}", flush=True)
                ok, extracted_answer = False, "No extracted answer"
            pass_list.append(1.0 if ok else 0.0)
            extracted.append(extracted_answer)
            judge_info.append(
                {
                    "rule_judge_result": bool(ok),
                    "finally_judge_result": bool(ok),
                    "extracted_answer": extracted_answer,
                    "fail_mode": "ok" if ok else "fail",
                }
            )
            token_id_lists.append(tokenizer.encode(text))
        think_lens = []
        ans_lens = []
        for text in case_texts:
            t, a = _think_answer_lens(text, tokenizer, _think_end_token(family))
            think_lens.append(t)
            ans_lens.append(a)
        if dataset in MATH_QA_DATASETS or dataset in MCQ_DATASETS:
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
            if dump_steps:
                rec["step_c"] = fm.step_values(meta.get("last_c"))
                rec["step_dc"] = fm.step_values(meta.get("penult_dc"))
                rec["step_dc_n3"] = fm.step_values(meta.get("n3_dc"))
                rec["step_dc_n4"] = fm.step_values(meta.get("n4_dc"))
                rec["step_match"] = fm.step_match_flags(meta.get("penult_match"))
            sample_deltas.append(rec)
        div = _diversity_summary(token_id_lists)
        row = _case_record(dataset, sample)
        row.update(
            {
                "idx": start + k,
                "completion": case_texts if args.save_completions else None,
                "extracted": extracted,
                "generated_tokens": case_toks,
                "avg_generated_tokens": sum(case_toks) / max(n, 1),
                "avg_think_tokens": sum(think_lens) / max(n, 1),
                "avg_answer_tokens": sum(ans_lens) / max(n, 1),
                "n": n,
                "judge_info": judge_info,
                "passat1": sum(pass_list) / max(n, 1),
                "passatk": max(pass_list) if pass_list else 0.0,
                "vote_acc": pass_list[first_index],
                "vote_key": voted,
                "vote_counts": {str(a): c for a, c in vote_counts.items()},
                "vote_acc_weighted": fm.weighted_vote_acc(pass_list, vote_keys, weights),
                "vote_acc_lo_ent": fm.vote_on_subset(pass_list, vote_keys, lo_ent_idx),
                "vote_acc_hi_logp": fm.vote_on_subset(pass_list, vote_keys, hi_logp_idx),
                "vote_acc_tail_c_weighted": paper["vote_acc_tail_c_weighted"],
                "vote_acc_tail_c_filter_20": paper["vote_acc_tail_c_filter_20"],
                "vote_acc_tail_c_filter_30": paper["vote_acc_tail_c_filter_30"],
                "vote_acc_tail_c_filter_50": paper["vote_acc_tail_c_filter_50"],
                "vote_acc_joint_dc_weighted": paper["vote_acc_joint_dc_weighted"],
                "joint_retained": paper["joint_retained"],
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
        results.append(row)

    judge_s = time.perf_counter() - t_judge0
    wall_s = time.perf_counter() - t_wall0
    total = len(results)
    n_tok = sum(int(x) for x in token_counts)

    def avg(key):
        return sum(r[key] for r in results) / total if total else 0.0

    stats = {
        "dataset": dataset,
        "data_path": data_path,
        "start_idx": start,
        "end_idx": end,
        "total_num": total,
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
        "vote_acc_tail_c_weighted": avg("vote_acc_tail_c_weighted"),
        "vote_acc_tail_c_filter_20": avg("vote_acc_tail_c_filter_20"),
        "vote_acc_tail_c_filter_30": avg("vote_acc_tail_c_filter_30"),
        "vote_acc_tail_c_filter_50": avg("vote_acc_tail_c_filter_50"),
        "vote_acc_joint_dc_weighted": avg("vote_acc_joint_dc_weighted"),
        "mean_joint_retained": avg("joint_retained"),
        "joint_rho": (
            float(args.joint_rho)
            if getattr(args, "joint_rho", None) is not None
            else agg.default_rho(family)
        ),
        "tail_mode": agg.resolve_tail_mode(family)[0],
        "vote_acc_weighted": avg("vote_acc_weighted"),
        "vote_acc_lo_ent": avg("vote_acc_lo_ent"),
        "vote_acc_hi_logp": avg("vote_acc_hi_logp"),
        "pass@1_high_ent": avg("passat1_high_ent"),
        "pass@k_high_ent": avg("passatk_high_ent"),
        "pass@1_low_ent": avg("passat1_low_ent"),
        "pass@k_low_ent": avg("passatk_low_ent"),
        "pass@1_hi_logp": avg("passat1_hi_logp"),
        "pass@k_hi_logp": avg("passatk_hi_logp"),
        "pass@1_hi_last_mp": fm.avg_or_none(results, "passat1_hi_last_mp"),
        "pass@k_hi_last_mp": fm.avg_or_none(results, "passatk_hi_last_mp"),
        "pass@1_hi_last_mcos": fm.avg_or_none(results, "passat1_hi_last_mcos"),
        "pass@k_hi_last_mcos": fm.avg_or_none(results, "passatk_hi_last_mcos"),
        "pass@1_hi_last_c": fm.avg_or_none(results, "passat1_hi_last_c"),
        "pass@k_hi_last_c": fm.avg_or_none(results, "passatk_hi_last_c"),
        "pass@1_hi_all_dmp": fm.avg_or_none(results, "passat1_hi_all_dmp"),
        "pass@k_hi_all_dmp": fm.avg_or_none(results, "passatk_hi_all_dmp"),
        "pass@1_hi_all_dmc": fm.avg_or_none(results, "passat1_hi_all_dmc"),
        "pass@k_hi_all_dmc": fm.avg_or_none(results, "passatk_hi_all_dmc"),
        "pass@1_hi_all_dc": fm.avg_or_none(results, "passat1_hi_all_dc"),
        "pass@k_hi_all_dc": fm.avg_or_none(results, "passatk_hi_all_dc"),
        "pass@1_hi_all_dc_n3": fm.avg_or_none(results, "passat1_hi_all_dc_n3"),
        "pass@k_hi_all_dc_n3": fm.avg_or_none(results, "passatk_hi_all_dc_n3"),
        "pass@1_hi_all_dc_n4": fm.avg_or_none(results, "passat1_hi_all_dc_n4"),
        "pass@k_hi_all_dc_n4": fm.avg_or_none(results, "passatk_hi_all_dc_n4"),
        "pass@1_hi_switch_dmp": fm.avg_or_none(results, "passat1_hi_switch_dmp"),
        "pass@k_hi_switch_dmp": fm.avg_or_none(results, "passatk_hi_switch_dmp"),
        "pass@1_hi_switch_dmc": fm.avg_or_none(results, "passat1_hi_switch_dmc"),
        "pass@k_hi_switch_dmc": fm.avg_or_none(results, "passatk_hi_switch_dmc"),
        "pass@1_hi_tail_c": fm.avg_or_none(results, "passat1_hi_tail_c"),
        "pass@k_hi_tail_c": fm.avg_or_none(results, "passatk_hi_tail_c"),
        "pass@1_hi_tail_dmp": fm.avg_or_none(results, "passat1_hi_tail_dmp"),
        "pass@k_hi_tail_dmp": fm.avg_or_none(results, "passatk_hi_tail_dmp"),
        "pass@1_hi_tail_dmc": fm.avg_or_none(results, "passat1_hi_tail_dmc"),
        "pass@k_hi_tail_dmc": fm.avg_or_none(results, "passatk_hi_tail_dmc"),
        "pass@1_hi_tail_dc": fm.avg_or_none(results, "passat1_hi_tail_dc"),
        "pass@k_hi_tail_dc": fm.avg_or_none(results, "passatk_hi_tail_dc"),
        "pass@1_hi_tail_dc_n3": fm.avg_or_none(results, "passat1_hi_tail_dc_n3"),
        "pass@k_hi_tail_dc_n3": fm.avg_or_none(results, "passatk_hi_tail_dc_n3"),
        "pass@1_hi_tail_dc_n4": fm.avg_or_none(results, "passat1_hi_tail_dc_n4"),
        "pass@k_hi_tail_dc_n4": fm.avg_or_none(results, "passatk_hi_tail_dc_n4"),
        "pass@1_hi_tail10_c": fm.avg_or_none(results, "passat1_hi_tail10_c"),
        "pass@k_hi_tail10_c": fm.avg_or_none(results, "passatk_hi_tail10_c"),
        "pass@1_hi_tail10_dmp": fm.avg_or_none(results, "passat1_hi_tail10_dmp"),
        "pass@k_hi_tail10_dmp": fm.avg_or_none(results, "passatk_hi_tail10_dmp"),
        "pass@1_hi_tail10_dmc": fm.avg_or_none(results, "passat1_hi_tail10_dmc"),
        "pass@k_hi_tail10_dmc": fm.avg_or_none(results, "passatk_hi_tail10_dmc"),
        "pass@1_hi_tail10_dc": fm.avg_or_none(results, "passat1_hi_tail10_dc"),
        "pass@k_hi_tail10_dc": fm.avg_or_none(results, "passatk_hi_tail10_dc"),
        "pass@1_hi_tail10_dc_n3": fm.avg_or_none(results, "passat1_hi_tail10_dc_n3"),
        "pass@k_hi_tail10_dc_n3": fm.avg_or_none(results, "passatk_hi_tail10_dc_n3"),
        "pass@1_hi_tail10_dc_n4": fm.avg_or_none(results, "passat1_hi_tail10_dc_n4"),
        "pass@k_hi_tail10_dc_n4": fm.avg_or_none(results, "passatk_hi_tail10_dc_n4"),
        "pass@1_hi_tail20_c": fm.avg_or_none(results, "passat1_hi_tail20_c"),
        "pass@k_hi_tail20_c": fm.avg_or_none(results, "passatk_hi_tail20_c"),
        "pass@1_hi_tail20_dmp": fm.avg_or_none(results, "passat1_hi_tail20_dmp"),
        "pass@k_hi_tail20_dmp": fm.avg_or_none(results, "passatk_hi_tail20_dmp"),
        "pass@1_hi_tail20_dmc": fm.avg_or_none(results, "passat1_hi_tail20_dmc"),
        "pass@k_hi_tail20_dmc": fm.avg_or_none(results, "passatk_hi_tail20_dmc"),
        "pass@1_hi_tail20_dc": fm.avg_or_none(results, "passat1_hi_tail20_dc"),
        "pass@k_hi_tail20_dc": fm.avg_or_none(results, "passatk_hi_tail20_dc"),
        "pass@1_hi_tail20_dc_n3": fm.avg_or_none(results, "passat1_hi_tail20_dc_n3"),
        "pass@k_hi_tail20_dc_n3": fm.avg_or_none(results, "passatk_hi_tail20_dc_n3"),
        "pass@1_hi_tail20_dc_n4": fm.avg_or_none(results, "passat1_hi_tail20_dc_n4"),
        "pass@k_hi_tail20_dc_n4": fm.avg_or_none(results, "passatk_hi_tail20_dc_n4"),
        "mean_last_mp": fm.mean_defined(r.get("mean_last_mp") for r in results),
        "mean_last_mcos": fm.mean_defined(r.get("mean_last_mcos") for r in results),
        "mean_last_c": fm.mean_defined(r.get("mean_last_c") for r in results),
        "mean_all_dmp": fm.mean_defined(r.get("mean_all_dmp") for r in results),
        "mean_all_dmc": fm.mean_defined(r.get("mean_all_dmc") for r in results),
        "mean_all_dc": fm.mean_defined(r.get("mean_all_dc") for r in results),
        "mean_all_dc_n3": fm.mean_defined(r.get("mean_all_dc_n3") for r in results),
        "mean_all_dc_n4": fm.mean_defined(r.get("mean_all_dc_n4") for r in results),
        "avg_token_length-all": avg("avg_generated_tokens"),
        "mean_unique_seq_ratio": (
            sum(r["diversity"].get("unique_ratio", 0.0) for r in results) / total if total else 0.0
        ),
        "mean_logprob": fm.mean_defined(
            x for r in results for x in (r.get("mean_logprob_list") or [])
        ),
        "mean_entropy": fm.mean_defined(
            x for r in results for x in (r.get("mean_entropy_list") or [])
        ),
        "generate_s": generate_s,
        "judge_s": judge_s,
        "wall_s": wall_s,
        "completion_tokens": n_tok,
        "tokens_per_s": (n_tok / generate_s) if generate_s > 0 else 0.0,
    }
    print(f"\n=== {dataset} statistics ===")
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
    return {"stats": stats, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Over-generate then filter / majority-vote")
    parser.add_argument(
        "--model_path",
        default=os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.5-4B"),
    )
    parser.add_argument("--family", default=None, choices=["qwen3", "qwen3.5", "qwen3.8", "gemma4", "generic"])
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
        help="Sampling top_k. Default 20 for Qwen, 64 for Gemma 4.",
    )
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=None,
        help="Default 1.5 for Qwen, 0.0 for Gemma 4.",
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
            json.dump({"args": vars(args), "prompt": prompt, "runs": payload}, f, ensure_ascii=False, indent=2)
        print(f"wrote {out}")
        if step_path:
            print(f"wrote step traces {step_path}")
        elif _dump_step_stats_enabled():
            print(
                "[ogf] WARNING: DUMP_STEP_STATS=1 but no .steps.jsonl "
                "(last_c / penult_dc lists were empty)",
                flush=True,
            )


if __name__ == "__main__":
    main()
