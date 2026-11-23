"""Last vs early-exit layer margins for overgen-filter.

Matches angular-sampling:

- Mp / Mc use last-layer softmax top-K ids (last-layer only; ΔMp/ΔMc are not stored)
- DeepConf C_i = -mean log P of that layer's own top-K (DEEPCONF_K, default 20)
- ΔC = C_last − C_early = mean_logp_early − mean_logp_last
- Early exits: n-2 (penult), n-3, n-4
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

MARGIN_TENSOR_KEYS = (
    "last_mp",
    "last_mcos",
    "last_c",
    "penult_dc",
    "penult_match",
    "n3_dc",
    "n4_dc",
)


def sliced_margin_kwargs(output, n: Optional[int] = None) -> dict:
    out = {}
    for key in MARGIN_TENSOR_KEYS:
        t = getattr(output, key, None)
        if t is None:
            continue
        if n is not None and hasattr(t, "shape") and t.shape and t.shape[0] >= n:
            t = t[:n]
        out[key] = t
    return out


def layer_margins_enabled() -> bool:
    return os.environ.get("OGF_LAYER_MARGINS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _lm_head_row_norms(lm_head) -> Optional[torch.Tensor]:
    cached = getattr(lm_head, "_ogf_row_norms", None)
    if cached is not None:
        return cached
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.ndim != 2:
        return None
    local = torch.linalg.vector_norm(weight.float(), dim=-1)
    tp_size = int(getattr(lm_head, "tp_size", 1) or 1)
    if tp_size > 1:
        from sglang.srt.distributed import tensor_model_parallel_all_gather

        local = tensor_model_parallel_all_gather(local)
    lm_head._ogf_row_norms = local
    return local


def _margins_on_ids(ell, hidden, ids, w_norms, probs):
    p_k = torch.nan_to_num(probs.gather(-1, ids), nan=0.0, posinf=0.0, neginf=0.0)
    logit_k = ell.gather(-1, ids)
    h_norm = torch.linalg.vector_norm(hidden, dim=-1).clamp_min(1e-12).unsqueeze(-1)
    w_k = w_norms.to(device=hidden.device, dtype=hidden.dtype)[ids].clamp_min(1e-12)
    cos_k = (logit_k / (w_k * h_norm)).clamp(-1.0, 1.0)
    mp = p_k[:, 0] - p_k[:, 1:].mean(dim=-1)
    mc = cos_k[:, 0] - cos_k[:, 1:].mean(dim=-1)
    return (
        torch.nan_to_num(mp, nan=0.0),
        torch.nan_to_num(mc, nan=0.0),
    )


def _own_top_k_mean_logp(probs: torch.Tensor, k_conf: int) -> torch.Tensor:
    return (
        torch.topk(probs, k=k_conf, dim=-1).values.clamp_min(1e-12).log().mean(dim=-1)
    )


def _early_dc(
    logits_processor,
    lm_head,
    meta: LogitsMetadata,
    hidden_early: Optional[torch.Tensor],
    last_top20_logp: torch.Tensor,
    k_conf: int,
    n_h: int,
) -> Optional[torch.Tensor]:
    """ΔC = mean_logp_early − mean_logp_last for one early-exit hidden."""
    if hidden_early is None:
        return None
    try:
        logits_e = logits_processor._get_logits(
            hidden_early, lm_head, meta, use_logits_buffer=False
        )
        probs_e = torch.softmax(logits_e.float()[:n_h], dim=-1)
        n = min(int(last_top20_logp.shape[0]), int(probs_e.shape[0]), n_h)
        if n <= 0:
            return None
        early_logp = _own_top_k_mean_logp(probs_e[:n], k_conf)
        return early_logp - last_top20_logp[:n]
    except Exception as e:
        logger.warning("attach_layer_margins early ΔC failed: %s: %s", type(e).__name__, e)
        return None


def attach_layer_margins(
    logits_out: LogitsProcessorOutput,
    logits_processor,
    lm_head,
    forward_batch: ForwardBatch,
    hidden_last: torch.Tensor,
    hidden_penult: Optional[torch.Tensor],
    hidden_n3: Optional[torch.Tensor] = None,
    hidden_n4: Optional[torch.Tensor] = None,
    topk_margin_k: int = 5,
    deepconf_k: int = 20,
) -> None:
    """Fill last-layer / Δ tensors on ``logits_out``. Decode only.

    penult (n-2): last Mp/Mc/C, ΔC, and top-1 match (ΔMp/ΔMc are not stored).
    n-3 / n-4: DeepConf ΔC only (C_last − C_early).
    """
    if not layer_margins_enabled():
        return
    if hidden_last is None or hidden_penult is None or not forward_batch.forward_mode.is_decode():
        return
    logits_last = logits_out.next_token_logits
    if logits_last is None:
        return
    w_norms = _lm_head_row_norms(lm_head)
    if w_norms is None:
        return
    try:
        meta = LogitsMetadata.from_forward_batch(forward_batch)
        logits_penult = logits_processor._get_logits(
            hidden_penult, lm_head, meta, use_logits_buffer=False
        )
    except Exception as e:
        logger.warning("attach_layer_margins penult logits failed: %s: %s", type(e).__name__, e)
        return

    ell_l = logits_last.float()
    ell_p = logits_penult.float()
    h_l = hidden_last.float()
    h_p = hidden_penult.float()
    if h_l.dim() == 3:
        h_l = h_l.reshape(-1, h_l.shape[-1])
    if h_p.dim() == 3:
        h_p = h_p.reshape(-1, h_p.shape[-1])
    n_h = min(int(ell_l.shape[0]), int(h_l.shape[0]), int(h_p.shape[0]), int(ell_p.shape[0]))
    k = min(max(int(topk_margin_k), 2), int(ell_l.shape[-1]))
    if n_h <= 0 or k < 2:
        return
    try:
        ids = torch.topk(ell_l[:n_h], k=k, dim=-1).indices
        probs_l = torch.softmax(ell_l[:n_h], dim=-1)
        probs_p = torch.softmax(ell_p[:n_h], dim=-1)
        mp_l, mc_l = _margins_on_ids(ell_l[:n_h], h_l[:n_h], ids, w_norms, probs_l)
        k_conf = min(
            max(int(os.getenv("DEEPCONF_K", str(deepconf_k)) or deepconf_k), 1),
            int(probs_l.shape[-1]),
        )
        last_top20_logp = _own_top_k_mean_logp(probs_l, k_conf)
        penult_top20_logp = _own_top_k_mean_logp(probs_p, k_conf)
        match = (ell_l[:n_h].argmax(dim=-1) == ell_p[:n_h].argmax(dim=-1)).to(
            dtype=last_top20_logp.dtype
        )
        logits_out.last_mp = mp_l
        logits_out.last_mcos = mc_l
        logits_out.last_c = -last_top20_logp
        # ΔMp / ΔMc intentionally not stored (C / ΔC only for early-exit deltas).
        logits_out.penult_dmp = None
        logits_out.penult_dmc = None
        logits_out.penult_dc = penult_top20_logp - last_top20_logp
        logits_out.penult_match = match
        logits_out.n3_dc = _early_dc(
            logits_processor, lm_head, meta, hidden_n3, last_top20_logp, k_conf, n_h
        )
        logits_out.n4_dc = _early_dc(
            logits_processor, lm_head, meta, hidden_n4, last_top20_logp, k_conf, n_h
        )
    except Exception as e:
        logger.warning("attach_layer_margins compute failed: %s: %s", type(e).__name__, e)
        return


def _ogf_hidden(wrapper, inner, name: str, n: Optional[int]):
    """Fetch ``ogf_<name>_hidden`` from model or wrapper."""
    meth = f"ogf_{name}_hidden"
    if inner is not None and hasattr(inner, meth):
        return getattr(inner, meth)(n)
    if hasattr(wrapper, meth):
        return getattr(wrapper, meth)(n)
    return None


def maybe_attach_layer_margins(
    logits_out,
    wrapper,
    forward_batch: ForwardBatch,
    hidden_last: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
) -> None:
    """Attach margins using wrapper.(logits_processor, lm_head, model)."""
    if logits_out is None or wrapper is None:
        return
    inner = getattr(wrapper, "model", None)
    lp = getattr(wrapper, "logits_processor", None)
    head = getattr(wrapper, "lm_head", None)
    if lp is None or head is None:
        return
    penult = _ogf_hidden(wrapper, inner, "penult", n)
    if hidden_last is None:
        hidden_last = _ogf_hidden(wrapper, inner, "last", n)
    hidden_n3 = _ogf_hidden(wrapper, inner, "n3", n)
    hidden_n4 = _ogf_hidden(wrapper, inner, "n4", n)
    attach_layer_margins(
        logits_out,
        lp,
        head,
        forward_batch,
        hidden_last,
        penult,
        hidden_n3=hidden_n3,
        hidden_n4=hidden_n4,
    )
