"""Trace metadata: DeepConf C, SGLang layer margins (ΔC), diversity."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

from confdyn import filter_metrics as fm

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
        f"DEEPCONF_K={os.environ.get('DEEPCONF_K', '20')} "
        f"OGF_LAYER_C_DEPTH={os.environ.get('OGF_LAYER_C_DEPTH', '1')} "
        f"OGF_LAYER_TOPK={os.environ.get('OGF_LAYER_TOPK', '0')}"
    )
    if n_c == 0 or n_dc == 0:
        print(
            f"[confdyn] ERROR: C/ΔC missing after first batch ({n} seqs, "
            f"last_c={n_c}/{n} penult_dc={n_dc}/{n} n3_dc={n_dc3}/{n} n4_dc={n_dc4}/{n}; "
            f"{flags}). mean_last_c_list / mean_all_dc_list will be null.",
            flush=True,
        )
        return
    print(
        f"[confdyn] C/ΔC present: last_c={n_c}/{n} (max {n_c_steps} steps) "
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

