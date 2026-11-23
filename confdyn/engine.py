"""SGLang engine construction and sampling-parameter helpers."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from confdyn.dump_io import _as_output_list
from confdyn.prompting import FAMILY_ENGINE, FAMILY_SAMPLING

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
    if getattr(args, "tree_search", False):
        # One generate() holds up to beam * branch live chunks per problem,
        # and never more than max_batch.
        beam_w = max(int(getattr(args, "tree_beam", 2) or 2), 1)
        branch_w = max(int(getattr(args, "tree_branch", 3) or 3), 1)
        init_w = max(int(getattr(args, "tree_init_branch", 0) or 0), 0)
        width = max(init_w, beam_w * branch_w)
        concurrent = width * n_cases
        cap = int(getattr(args, "max_batch", 0) or 0)
        if cap > 0:
            concurrent = min(concurrent, cap)
    else:
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
    for _name in (
        "linear_attn_backend",
        "linear_attn_decode_backend",
        "linear_attn_prefill_backend",
    ):
        _val = getattr(args, _name, None)
        if _val:
            kwargs[_name] = _val
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


def _expand_sample_major_images(
    case_images: List[Optional[List[Any]]], n_samples: int
) -> Optional[List[Any]]:
    """Mirror ``_expand_sample_major`` for per-case image lists (or None)."""
    if not case_images or all(imgs is None for imgs in case_images):
        return None
    flat: List[Any] = []
    for sample_idx in range(n_samples):
        del sample_idx
        for imgs in case_images:
            flat.append(imgs if imgs is not None else [])
    return flat


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
    image_data: Optional[List[Any]] = None,
) -> List[dict]:
    kw = dict(return_logprob=bool(args.return_logprob))
    if args.return_logprob and args.top_logprobs_num > 0:
        kw["top_logprobs_num"] = args.top_logprobs_num
    if image_data is not None:
        kw["image_data"] = image_data
    try:
        outputs = llm.generate(prompts, sp, **kw)
    except TypeError:
        if image_data is not None:
            raise
        outputs = llm.generate(prompts, sp)
    return _as_output_list(outputs)


def _generate_ids(llm, input_ids: List[List[int]], sp: Any, args) -> List[dict]:
    """Continue from token ids. ``input_ids`` is a batch of full prefixes."""
    kw = dict(input_ids=input_ids, return_logprob=bool(args.return_logprob))
    if args.return_logprob and args.top_logprobs_num > 0:
        kw["top_logprobs_num"] = args.top_logprobs_num
    outputs = llm.generate(sampling_params=sp, **kw)
    return _as_output_list(outputs)

