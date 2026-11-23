"""Optional per-step logprob / margin sidecars for dump JSON."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

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
    "step_top1_logp",
    "step_penult_top1_logp",
    "step_last_rank_logp",
    "step_penult_rank_logp",
    "step_penult_on_last_logp",
    "step_layer_c",
    "step_layer_top20",
    "step_layer_overlap",
    "step_sampled_token",
    "step_layer_top20_logit",
    "step_layer_top20_prob",
    "step_layer_top20_cos",
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

