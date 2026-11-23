"""Last vs early-exit layer margins for overgen-filter.

Matches angular-sampling:

- Mp / Mc use last-layer softmax top-K ids (last-layer only; ΔMp/ΔMc are not stored)
- DeepConf C_i = -mean log P of that layer's own top-K (DEEPCONF_K, default 20)
- ΔC = C_last − C_early = mean_logp_early − mean_logp_last
- Early exits: n-2 (penult), n-3, n-4
- Optional per-step C for the last ``OGF_LAYER_C_DEPTH`` layers (default 1).
  Depth 10 stores C at L-9 .. L. L is ``last_c``; L-1 .. L-9 are ``layer_c_1`` .. ``layer_c_9``.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# Last layer plus nine earlier layers. OGF_LAYER_C_DEPTH is clamped to this.
LAYER_C_MAX_DEPTH = 10

MARGIN_TENSOR_KEYS = (
    "last_mp",
    "last_mcos",
    "last_c",
    "penult_dc",
    "penult_match",
    # log p_L(v*) and log p_{L-1}(v*), v* = argmax of the last layer.
    # These stay exact when v* is outside the penultimate top-20.
    "top1_logp",
    "penult_top1_logp",
    "n3_dc",
    "n4_dc",
    # C at L-1 .. L-9. Populated only when OGF_LAYER_C_DEPTH > 1.
    *(f"layer_c_{offset}" for offset in range(1, LAYER_C_MAX_DEPTH)),
)

# Own top-DEEPCONF_K token ids. layer_top_0 is the last layer; layer_top_k is L-k.
# Filled only when OGF_LAYER_TOPK=1. Kept out of MARGIN_TENSOR_KEYS because each
# row is a length-k vector, not a scalar.
MARGIN_ID_KEYS = (
    "layer_top_0",
    "layer_top_1",
    "layer_top_2",
    "layer_top_3",
    "layer_top_4",
    "layer_top_5",
)

# Same rows as MARGIN_ID_KEYS, in rank order. logit and prob are the raw
# pre-temperature softmax. cos is cos θ in z_i = ||h|| ||W_i|| cos θ_i.
MARGIN_TOP_STAT_KEYS = tuple(
    f"layer_top_{stat}_{offset}"
    for stat in ("logit", "prob", "cos")
    for offset in range(6)
)

# Length-20 log-prob rows. Filled only when OGF_LAYER_RANKLOGP=1.
MARGIN_RANK_LOGP_KEYS = (
    "last_rank_logp",
    "penult_rank_logp",
    "penult_on_last_logp",
)


def snapshot_owned_margin_tensors(logits_out) -> None:
    """Copy margin tensors onto storage this process owns.

    Decode CUDA graph replay reuses its private pool. A top-k id or prob
    tensor left there can alias a later float kernel, so dumped ids become
    huge integers and probs leave ``[0, 1]``. ``clone`` while the values are
    still the top-k result, and again after replay so the returned object is
    not the pool buffer.
    """
    if logits_out is None:
        return
    for key in MARGIN_TENSOR_KEYS + MARGIN_ID_KEYS + MARGIN_TOP_STAT_KEYS + MARGIN_RANK_LOGP_KEYS:
        tensor = getattr(logits_out, key, None)
        if isinstance(tensor, torch.Tensor):
            setattr(logits_out, key, tensor.detach().clone())


def sliced_margin_kwargs(output, n: Optional[int] = None) -> dict:
    out = {}
    for key in MARGIN_TENSOR_KEYS:
        t = getattr(output, key, None)
        if t is None:
            continue
        if n is not None and hasattr(t, "shape") and t.shape and t.shape[0] >= n:
            t = t[:n]
        out[key] = t
    for key in MARGIN_ID_KEYS + MARGIN_TOP_STAT_KEYS + MARGIN_RANK_LOGP_KEYS:
        t = getattr(output, key, None)
        if t is None:
            continue
        if n is not None and hasattr(t, "shape") and t.shape and t.shape[0] >= n:
            t = t[:n]
        out[key] = t
    return out


def layer_c_depth() -> int:
    """How many final layers keep per-step DeepConf C. 1 is the last layer only."""
    raw = os.environ.get("OGF_LAYER_C_DEPTH", "1").strip()
    try:
        depth = int(raw)
    except ValueError:
        depth = 1
    return max(1, min(depth, LAYER_C_MAX_DEPTH))


def _env_off(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("0", "false", "no", "off", "")


def layer_topk_ids_only() -> bool:
    """Store token ids only. Skip logit, prob, and cos θ vectors."""
    return not _env_off("OGF_LAYER_TOPK_IDS_ONLY")


def layer_dump_table() -> bool:
    """Skip the top-20 logit vectors. Everything else is still stored.

    OGF_LAYER_DUMP=table keeps candidates, prob, cos θ, per-layer C, overlap,
    step C / ΔC / match, and the decoded token. Unset keeps logits too.
    IDS_ONLY runs are unchanged.
    """
    return os.environ.get("OGF_LAYER_DUMP", "").strip().lower() == "table"


def _env_k(name: str, default: int, vocab: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        k = int(raw) if raw else int(default)
    except ValueError:
        k = int(default)
    return min(max(k, 1), max(int(vocab), 1))


def layer_rank_logp_enabled() -> bool:
    """Store the 20 log-probs used to split ΔC by rank and to score suppression.

    Three vectors, each length 20, in rank order of the last layer unless noted:
    last_rank_logp is layer L's own top-20. penult_rank_logp is layer L-1's own
    top-20. penult_on_last_logp is p_{L-1} of layer L's top-20 token ids.
    """
    return os.environ.get("OGF_LAYER_RANKLOGP", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def layer_topk_enabled() -> bool:
    """Record each layer's own top-DEEPCONF_K token ids for overlap stats."""
    return os.environ.get("OGF_LAYER_TOPK", "0").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def layer_margins_enabled() -> bool:
    return os.environ.get("OGF_LAYER_MARGINS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def layer_dc_enabled() -> bool:
    """Compute penultimate-layer ΔC. Default on.

    Off keeps last-layer C and skips the extra penultimate lm_head.
    """
    return os.environ.get("OGF_LAYER_DC", "1").strip().lower() not in (
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


def _own_top_k_ids(probs: torch.Tensor, k_conf: int, n_h: int) -> torch.Tensor:
    kk = min(int(k_conf), int(probs.shape[-1]))
    return torch.topk(probs[:n_h], k=kk, dim=-1).indices


def _assign_top_ids(logits_out, offset: int, probs: torch.Tensor, k: int) -> None:
    """Store this layer's own top-k token ids, without logit, prob, or cos θ."""
    kk = min(int(k), int(probs.shape[-1]))
    if kk < 1:
        return
    setattr(logits_out, f"layer_top_{offset}", torch.topk(probs, k=kk, dim=-1).indices)


def _assign_top_decomp(
    logits_out,
    offset: int,
    logits: torch.Tensor,
    hidden: torch.Tensor,
    probs: torch.Tensor,
    w_norms: torch.Tensor,
    k_conf: int,
) -> None:
    """Store this layer's own top-k ids plus logit, prob, and cos θ.

    ``logits`` / ``hidden`` / ``probs`` are already the decode rows. cos θ is
    the kernel logit with both norms removed: z_i / (||h|| ||W_i||).
    """
    kk = min(int(k_conf), int(probs.shape[-1]))
    if kk < 1:
        return
    ids = torch.topk(probs, k=kk, dim=-1).indices
    logit_k = logits.gather(-1, ids)
    prob_k = probs.gather(-1, ids)
    h_norm = torch.linalg.vector_norm(hidden, dim=-1).clamp_min(1e-12).unsqueeze(-1)
    w_k = w_norms.to(device=ids.device, dtype=logit_k.dtype)[ids].clamp_min(1e-12)
    cos_k = (logit_k / (w_k * h_norm)).clamp(-1.0, 1.0)
    setattr(logits_out, f"layer_top_{offset}", ids)
    if not layer_dump_table():
        setattr(logits_out, f"layer_top_logit_{offset}", logit_k)
    setattr(logits_out, f"layer_top_prob_{offset}", prob_k)
    setattr(logits_out, f"layer_top_cos_{offset}", cos_k)


def _own_top_k_mean_logp(probs: torch.Tensor, k_conf: int) -> torch.Tensor:
    return (
        torch.topk(probs, k=k_conf, dim=-1).values.clamp_min(1e-12).log().mean(dim=-1)
    )


def _attach_last_c_only(
    logits_out: LogitsProcessorOutput,
    lm_head,
    hidden_last: torch.Tensor,
    logits_last: torch.Tensor,
    topk_margin_k: int,
    deepconf_k: int,
) -> None:
    """Last-layer C from logits the sampler already has. No penultimate matmul."""
    w_norms = _lm_head_row_norms(lm_head)
    if w_norms is None:
        return
    ell_l = logits_last.float()
    h_l = hidden_last.float()
    if h_l.dim() == 3:
        h_l = h_l.reshape(-1, h_l.shape[-1])
    n_h = min(int(ell_l.shape[0]), int(h_l.shape[0]))
    k = min(max(int(topk_margin_k), 2), int(ell_l.shape[-1]))
    if n_h <= 0 or k < 2:
        return
    ids = torch.topk(ell_l[:n_h], k=k, dim=-1).indices
    probs_l = torch.softmax(ell_l[:n_h], dim=-1)
    mp_l, mc_l = _margins_on_ids(ell_l[:n_h], h_l[:n_h], ids, w_norms, probs_l)
    k_conf = min(
        max(int(os.getenv("DEEPCONF_K", str(deepconf_k)) or deepconf_k), 1),
        int(probs_l.shape[-1]),
    )
    last_top20_logp = _own_top_k_mean_logp(probs_l, k_conf)
    v_star = ell_l[:n_h].argmax(dim=-1, keepdim=True)
    top1_logp = probs_l.gather(-1, v_star).clamp_min(1e-30).log().squeeze(-1)
    logits_out.last_mp = mp_l
    logits_out.last_mcos = mc_l
    logits_out.last_c = -last_top20_logp
    logits_out.penult_dmp = None
    logits_out.penult_dmc = None
    logits_out.penult_dc = None
    logits_out.penult_match = None
    logits_out.top1_logp = top1_logp
    logits_out.penult_top1_logp = None
    logits_out.n3_dc = None
    logits_out.n4_dc = None
    snapshot_owned_margin_tensors(logits_out)


def attach_layer_margins(
    logits_out: LogitsProcessorOutput,
    logits_processor,
    lm_head,
    forward_batch: ForwardBatch,
    hidden_last: torch.Tensor,
    hidden_penult: Optional[torch.Tensor],
    topk_margin_k: int = 5,
    deepconf_k: int = 20,
    extra_hiddens: Optional[list] = None,
) -> None:
    """Fill last-layer C and, unless OGF_LAYER_DC=0, n-2 ΔC. Decode only."""
    if not layer_margins_enabled():
        return
    if hidden_last is None or not forward_batch.forward_mode.is_decode():
        return
    logits_last = logits_out.next_token_logits
    if logits_last is None:
        return
    if hidden_penult is None or not layer_dc_enabled():
        try:
            _attach_last_c_only(
                logits_out,
                lm_head,
                hidden_last,
                logits_last,
                topk_margin_k,
                deepconf_k,
            )
        except Exception as e:
            logger.warning("attach_layer_margins last-C failed: %s: %s", type(e).__name__, e)
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
        v_star = ell_l[:n_h].argmax(dim=-1, keepdim=True)
        # Floor only so log is finite. 1e-30 is below any top-20 mass we keep.
        top1_logp = probs_l.gather(-1, v_star).clamp_min(1e-30).log().squeeze(-1)
        penult_top1_logp = probs_p.gather(-1, v_star).clamp_min(1e-30).log().squeeze(-1)
        match = (v_star.squeeze(-1) == ell_p[:n_h].argmax(dim=-1)).to(
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
        logits_out.top1_logp = top1_logp
        logits_out.penult_top1_logp = penult_top1_logp
        if layer_rank_logp_enabled():
            k_rank = min(20, int(probs_l.shape[-1]))
            last_ids = torch.topk(probs_l, k=k_rank, dim=-1).indices
            penult_ids = torch.topk(probs_p, k=k_rank, dim=-1).indices
            floor = 1e-30
            logits_out.last_rank_logp = probs_l.gather(-1, last_ids).clamp_min(floor).log()
            logits_out.penult_rank_logp = probs_p.gather(-1, penult_ids).clamp_min(floor).log()
            # Same token ids as last_ids, read at the penultimate layer.
            logits_out.penult_on_last_logp = probs_p.gather(-1, last_ids).clamp_min(floor).log()
        # Filled below when that early layer's hidden is stashed.
        # n3 is L-2 (third from the end): C_last - C_{L-2}.
        logits_out.n3_dc = None
        logits_out.n4_dc = None
        record_ids = layer_topk_enabled()
        if record_ids and layer_topk_ids_only():
            # Last layer keeps its top-1 id; the penultimate layer keeps top-3.
            # C / ΔC still use DEEPCONF_K. Logit, prob, and cos θ are not stored.
            vocab = int(probs_l.shape[-1])
            _assign_top_ids(
                logits_out, 0, probs_l, _env_k("OGF_LAYER_TOPK_LAST_K", 1, vocab)
            )
            _assign_top_ids(
                logits_out, 1, probs_p, _env_k("OGF_LAYER_TOPK_PENULT_K", 3, vocab)
            )
        elif record_ids:
            # Token-id dump can be shorter than the C window. Unset keeps
            # DEEPCONF_K, so existing top-k dumps stay the same length.
            k_ids = k_conf
            raw_ids = os.getenv("OGF_LAYER_TOPK_K", "").strip()
            if raw_ids:
                try:
                    k_ids = min(max(int(raw_ids), 1), int(probs_l.shape[-1]))
                except ValueError:
                    k_ids = k_conf
            _assign_top_decomp(
                logits_out, 0, ell_l[:n_h], h_l[:n_h], probs_l, w_norms, k_ids
            )
            _assign_top_decomp(
                logits_out, 1, ell_p[:n_h], h_p[:n_h], probs_p, w_norms, k_ids
            )
        if layer_c_depth() > 1:
            # C at L-1. Same top-k logp already used for penult ΔC.
            logits_out.layer_c_1 = -penult_top20_logp
            for offset, hidden in enumerate(extra_hiddens or [], start=2):
                if hidden is None or offset >= layer_c_depth():
                    continue
                h = hidden.float()
                if h.dim() == 3:
                    h = h.reshape(-1, h.shape[-1])
                if int(h.shape[0]) < n_h:
                    continue
                try:
                    logits_early = logits_processor._get_logits(
                        hidden, lm_head, meta, use_logits_buffer=False
                    )
                except Exception as early_err:
                    logger.warning(
                        "attach_layer_margins layer L-%s logits failed: %s",
                        offset,
                        early_err,
                    )
                    continue
                probs_e = torch.softmax(logits_early.float()[:n_h], dim=-1)
                early_logp = _own_top_k_mean_logp(probs_e, k_conf)
                setattr(logits_out, f"layer_c_{offset}", -early_logp)
                # Same definition as penult_dc: C_last - C_early.
                if offset == 2:
                    logits_out.n3_dc = early_logp - last_top20_logp
                elif offset == 3:
                    logits_out.n4_dc = early_logp - last_top20_logp
                if record_ids and offset <= 5 and not layer_topk_ids_only():
                    _assign_top_decomp(
                        logits_out,
                        offset,
                        logits_early.float()[:n_h],
                        h[:n_h],
                        probs_e,
                        w_norms,
                        k_conf,
                    )
        snapshot_owned_margin_tensors(logits_out)
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
    penult = _ogf_hidden(wrapper, inner, "penult", n) if layer_dc_enabled() else None
    if hidden_last is None:
        hidden_last = _ogf_hidden(wrapper, inner, "last", n)
    extra = None
    depth = layer_c_depth()
    if (
        layer_dc_enabled()
        and depth > 1
        and inner is not None
        and hasattr(inner, "ogf_early_hidden")
    ):
        extra = [inner.ogf_early_hidden(offset, n) for offset in range(2, depth)]
    attach_layer_margins(
        logits_out,
        lp,
        head,
        forward_batch,
        hidden_last,
        penult,
        extra_hiddens=extra,
    )
