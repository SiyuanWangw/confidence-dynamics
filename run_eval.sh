#!/usr/bin/env bash
# Sample a dataset and write one dump. Table 1 uses scripts/run_main_table.sh,
# which calls this with N=50.
#
#   MODELS=Qwen3-8B DATASETS=hmmt NUM_SAMPLES=50 bash run_eval.sh
#   MODELS=NVIDIA-Nemotron-3-Nano-4B-BF16 DATASETS=gpqa_diamond NUM_SAMPLES=50 bash run_eval.sh
#
# Decoding follows Appendix D.2 unless TEMPERATURE / TOP_K / PRESENCE_PENALTY
# / MAX_NEW_TOKENS are set:
#   Qwen3:    T=0.6 top_k=20 top_p=0.95 presence_penalty=1.5
#             Countdown6 uses T=1.0
#   Nemotron: T=1.0 top_k off (top_k=-1) top_p=0.95 presence_penalty=0
#   HMMT and AIME: 40,000 new tokens. Other Table 1 datasets: 32,768.
# Length-capped traces are dropped unless KEEP_TRUNCATED=1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-warning}"
export DUMP_STEP_STATS="${DUMP_STEP_STATS:-1}"
export OGF_LAYER_MARGINS="${OGF_LAYER_MARGINS:-1}"
# 1 = last-layer C only. 10 stores per-step C for the last ten layers.
export OGF_LAYER_C_DEPTH="${OGF_LAYER_C_DEPTH:-1}"
# 1 records each stored layer's own top-DEEPCONF_K token ids and the overlap
# of that set with the last layer's top-k. Default off; the lists are long.
export OGF_LAYER_TOPK="${OGF_LAYER_TOPK:-0}"
export DEEPCONF_K="${DEEPCONF_K:-20}"
export LOCAL_TAIL_WINDOW="${LOCAL_TAIL_WINDOW:-2048}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
unset NCCL_DEBUG_SUBSYS || true
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"

# Prefer system libcuda / libnccl over pip nvidia-nccl (same as overgen-filter/ogf).
if [[ -d /usr/lib/x86_64-linux-gnu ]]; then
  _ld="${LD_LIBRARY_PATH:-}"
  _ld="${_ld//:\/usr\/lib\/x86_64-linux-gnu/}"
  _ld="${_ld#/usr/lib/x86_64-linux-gnu:}"
  _ld="${_ld#/usr/lib/x86_64-linux-gnu}"
  if [[ -n "$_ld" ]]; then
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${_ld}"
  else
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"
  fi
fi

PY="${CONFDYN_PYTHON:-${OGF_PYTHON:-python}}"
# Resolve python to an absolute path when possible (for conda prefix discovery).
if command -v "$PY" >/dev/null 2>&1; then
  PY="$(command -v "$PY")"
fi

# CUDA graph JIT needs C++20 <concepts>. Host g++-9 (Ubuntu 20.04) is too old.
# Prefer system g++-10..13; else conda-forge g++ in the active env (gxx_linux-64).
if [[ -z "${CXX:-}" ]]; then
  for _cxx in /usr/bin/g++-13 /usr/bin/g++-12 /usr/bin/g++-11 /usr/bin/g++-10; do
    if [[ -x "$_cxx" ]]; then
      _ver="${_cxx##*-}"
      export CC="${CC:-/usr/bin/gcc-${_ver}}"
      export CXX="$_cxx"
      break
    fi
  done
  _py_root="$(cd "$(dirname "$PY")/.." && pwd 2>/dev/null || true)"
  for _root in "${CONDA_PREFIX:-}" "$_py_root"; do
    if [[ -z "${CXX:-}" && -n "$_root" && -x "${_root}/bin/x86_64-conda-linux-gnu-g++" ]]; then
      export CC="${_root}/bin/x86_64-conda-linux-gnu-gcc"
      export CXX="${_root}/bin/x86_64-conda-linux-gnu-g++"
    fi
  done
fi
if [[ -n "${CXX:-}" ]]; then
  export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
fi
echo "[confdyn] host CXX=${CXX:-unset} CUDAHOSTCXX=${CUDAHOSTCXX:-unset}"
echo "[confdyn] OGF_LAYER_C_DEPTH=${OGF_LAYER_C_DEPTH} OGF_LAYER_DC=${OGF_LAYER_DC:-1} OGF_LAYER_TOPK=${OGF_LAYER_TOPK} OGF_LAYER_TOPK_IDS_ONLY=${OGF_LAYER_TOPK_IDS_ONLY:-0} OGF_LAYER_TOPK_LAST_K=${OGF_LAYER_TOPK_LAST_K:-} OGF_LAYER_TOPK_PENULT_K=${OGF_LAYER_TOPK_PENULT_K:-} DEEPCONF_K=${DEEPCONF_K}"
echo "[confdyn] LD_LIBRARY_PATH starts with ${LD_LIBRARY_PATH%%:*}"

MODELS_DIR="${MODELS_DIR:-${ROOT}/../models}"
MODELS="${MODELS:-Qwen3-8B}"
FAMILY="${FAMILY:-}"
DATASETS="${DATASETS:-hmmt}"
OUT_TAG="${OUT_TAG:-}"
# Per-trace seed is random_seed + case*N + sample. A second N=10 pass must
# use a different base (at least N) so its traces do not repeat the first pass.
RANDOM_SEED="${RANDOM_SEED:-0}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-1000}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
FILTER_K="${FILTER_K:-5}"
# One generate() must not hold the whole dataset. Early decode runs every
# live sequence in one batch, and the Qwen3 C/ΔC hook materializes
# [batch, vocab] logits. gpqa is 198*N and countdown is 300*N; that batch
# OOBs the GPU and NCCL reports illegal memory access. HMMT is 60*N;
# aime is 90*N (AIME 2024+2025+2026).
# Multimodal (MMMU-Pro) prefills vision features per request; text-default 512
# OOMs easily and NCCL then only shows Broken-pipe teardown on the other ranks.
_mm_ds=0
case " ${DATASETS} " in *" mmmu_pro150 "*) _mm_ds=1 ;; esac
if [[ -n "${MAX_BATCH+x}" ]]; then
  : # caller set MAX_BATCH explicitly
elif [[ "$_mm_ds" -eq 1 ]]; then
  MAX_BATCH="${MAX_BATCH_MM:-16}"
else
  MAX_BATCH=512
fi
TOP_LOGPROBS_NUM="${TOP_LOGPROBS_NUM:-0}"
JOINT_RHO="${JOINT_RHO:-}"

_cvd="${CUDA_VISIBLE_DEVICES:-}"
_cvd="${_cvd// /}"
if [[ -z "$_cvd" ]]; then _NGPU=1
else _commas="${_cvd//[^,]}"; _NGPU=$((${#_commas} + 1)); fi
# Default: tensor-parallel across all visible GPUs; DP=1.
TP_SIZE="${TP_SIZE:-$_NGPU}"
DP_SIZE="${DP_SIZE:-1}"

detect_family() {
  local name="$1"
  local blob
  blob="$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr '_' '.')"
  if [[ "$blob" == *gemma-4* || "$blob" == *gemma4* ]]; then echo gemma4
  elif [[ "$blob" == *nemotron* ]]; then echo nemotron3
  elif [[ "$blob" == *qwen3.8* || "$blob" == *qwen3-8* ]]; then echo qwen3.8
  elif [[ "$blob" == *qwen3.5* ]]; then echo qwen3.5
  elif [[ "$blob" == *qwen3* ]]; then echo qwen3
  else echo generic
  fi
}

max_tokens_for() {
  local ds="$1"
  case "$ds" in
    hmmt|aime) echo 40000 ;;
    # Qwen3 card: programming contests use 38912 output tokens.
    lcb_v6) echo 38912 ;;
    gsm8k) echo 8192 ;;
    hle300|gpqa_diamond|sgpqa200|olympiad|countdown6|countdown6all|mmmu_pro150) echo 32768 ;;
    *) echo 32768 ;;
  esac
}

sampling_for() {
  # sets TEMP TOPK PRESENCE via namerefs-like globals
  local fam="$1" ds="$2"
  # LiveCodeBench uses each model's own card, not the paper's Qwen
  # presence_penalty=1.5. Qwen3 thinking: T=0.6 top_k=20 top_p=0.95 min_p=0.
  # Nemotron reasoning: T=1.0 top_p=0.95, top_k off, no presence penalty.
  if [[ "$ds" == lcb_v6 ]]; then
    if [[ "$fam" == nemotron3 ]]; then
      TEMP="${TEMPERATURE:-1.0}"; TOPK="${TOP_K:--1}"; PRESENCE="${PRESENCE_PENALTY:-0.0}"
    elif [[ "$fam" == gemma4 ]]; then
      TEMP="${TEMPERATURE:-1.0}"; TOPK="${TOP_K:-64}"; PRESENCE="${PRESENCE_PENALTY:-0.0}"
    else
      TEMP="${TEMPERATURE:-0.6}"; TOPK="${TOP_K:-20}"; PRESENCE="${PRESENCE_PENALTY:-0}"
    fi
    return
  fi
  if [[ "$fam" == gemma4 ]]; then
    TEMP="${TEMPERATURE:-1.0}"; TOPK="${TOP_K:-64}"; PRESENCE="${PRESENCE_PENALTY:-0.0}"
  elif [[ "$fam" == nemotron3 ]]; then
    # Reasoning card: temperature 1.0, top_p 0.95. top_k=-1 keeps the full vocab.
    TEMP="${TEMPERATURE:-1.0}"; TOPK="${TOP_K:--1}"; PRESENCE="${PRESENCE_PENALTY:-0.0}"
  else
    TEMP="${TEMPERATURE:-0.6}"; TOPK="${TOP_K:-20}"; PRESENCE="${PRESENCE_PENALTY:-1.5}"
    if [[ "$ds" == countdown6 || "$ds" == countdown6all ]]; then
      TEMP="${TEMPERATURE:-1.0}"
    fi
  fi
}

mkdir -p "${ROOT}/results"
any_fail=0

for model in $MODELS; do
  MODEL_PATH="${MODELS_DIR}/${model}"
  if [[ ! -e "$MODEL_PATH" ]]; then
    echo "[confdyn] missing model: $MODEL_PATH" >&2
    exit 1
  fi
  fam="${FAMILY:-$(detect_family "$model")}"
  rho_args=()
  if [[ -n "$JOINT_RHO" ]]; then
    rho_args=(--joint_rho "$JOINT_RHO")
  fi

  for ds in $DATASETS; do
    sampling_for "$fam" "$ds"
    MT="$(max_tokens_for "$ds")"
    if [[ -n "${MAX_NEW_TOKENS:-}" ]]; then
      MT="$MAX_NEW_TOKENS"
    fi
    # max_new_tokens is the completion budget, and the prompt counts against
    # the same context. Qwen3-8B-Base is 32768; instruct Qwen3-8B is 40960,
    # so HMMT's 40000 only fits the instruct checkpoint. A 1024-token margin
    # covers HMMT. GPQA has a 5289-token prompt, so that margin is measured
    # when 1024 is not enough.
    if [[ -f "$MODEL_PATH/config.json" ]]; then
      ctx="$("$PY" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("max_position_embeddings") or 0))' "$MODEL_PATH/config.json" 2>/dev/null || echo 0)"
      if [[ "$ctx" =~ ^[0-9]+$ && "$ctx" -gt 1024 && "$MT" -gt $((ctx - 1024)) ]]; then
        prompt_max="$("$PY" - "$MODEL_PATH" "$fam" "$ds" "${START_IDX:-0}" "${END_IDX:-1000}" << 'PY'
import sys
from transformers import AutoTokenizer
from confdyn.data import _chat_prompt_and_images, _dataset_defaults, _load_samples
model, fam, ds, start, end = sys.argv[1:]
rows = _load_samples(_dataset_defaults(ds))[int(start):int(end)]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
n = 0
for sample in rows:
    prompt, _ = _chat_prompt_and_images(ds, sample, tok, fam)
    n = max(n, len(tok.encode(prompt)))
print(n)
PY
)"
        if [[ ! "$prompt_max" =~ ^[0-9]+$ ]]; then
          echo "[confdyn] could not measure prompt length for $ds" >&2
          any_fail=1
          continue
        fi
        # Match sglang's check: input tokens + max_new_tokens <= context.
        need=$((prompt_max + 8))
        if (( need <= 1024 )); then
          echo "[confdyn] max_new_tokens $MT does not fit context $ctx, capping to $((ctx - 1024))" >&2
          MT=$((ctx - 1024))
        elif (( need >= ctx )); then
          echo "[confdyn] longest $ds prompt is $prompt_max tokens, context is $ctx" >&2
          any_fail=1
          continue
        else
          echo "[confdyn] longest $ds prompt is $prompt_max tokens, capping max_new_tokens $MT to $((ctx - need)) (context $ctx)" >&2
          MT=$((ctx - need))
        fi
      fi
    fi
    tag_suffix=""
    [[ -n "$OUT_TAG" ]] && tag_suffix="_${OUT_TAG}"
    # Nonzero seed is a second pass. Put it in the filename so it cannot
    # replace the seed-0 dump used for the first N=10.
    if [[ "$RANDOM_SEED" != "0" ]]; then
      tag_suffix="${tag_suffix}_seed${RANDOM_SEED}"
    fi
    if [[ "${TREE_SEARCH:-0}" == "1" ]]; then
      _tree_score="${TREE_SCORE:-dc}"
      _tree_keep="${TREE_KEEP:-score}"
      # lastall: the final round keeps every chunk, not only the beam.
      # score does this by itself. A keep rule does it only with TREE_LAST_ALL=1.
      # wN: first round is N tokens. tailN: rank by the running tail, not the chunk mean.
      if [[ "$_tree_keep" != "score" ]]; then
        tag_suffix="${tag_suffix}_ktree_b${TREE_BEAM:-2}k${TREE_BRANCH:-3}c${TREE_CHUNK:-500}_keep${_tree_keep}"
      elif [[ "$_tree_score" == "c" ]]; then
        tag_suffix="${tag_suffix}_ctree_b${TREE_BEAM:-2}k${TREE_BRANCH:-3}c${TREE_CHUNK:-500}"
      elif [[ "$_tree_score" == "random" ]]; then
        tag_suffix="${tag_suffix}_rtree_b${TREE_BEAM:-2}k${TREE_BRANCH:-3}c${TREE_CHUNK:-500}"
      else
        tag_suffix="${tag_suffix}_dctree_b${TREE_BEAM:-2}k${TREE_BRANCH:-3}c${TREE_CHUNK:-500}"
      fi
      if [[ "${TREE_WARMUP:-0}" != "0" ]]; then
        tag_suffix="${tag_suffix}_w${TREE_WARMUP}"
      fi
      if [[ "${TREE_RANK:-chunk}" == "tail" ]]; then
        tag_suffix="${tag_suffix}_tail${TREE_TAIL:-2048}"
      fi
      if [[ "${TREE_INIT_BRANCH:-0}" != "0" ]]; then
        tag_suffix="${tag_suffix}_i${TREE_INIT_BRANCH}"
      fi
      if [[ "$_tree_keep" == "score" || "${TREE_LAST_ALL:-0}" == "1" ]]; then
        tag_suffix="${tag_suffix}_lastall"
      fi
    fi
    # Ordinary N sampling drops length-capped traces unless this is set.
    # The tag keeps those dumps from replacing a run that dropped them.
    if [[ "${KEEP_TRUNCATED:-0}" == "1" ]]; then
      tag_suffix="${tag_suffix}_keeptrunc"
    fi
    out_dir="${ROOT}/results/${ds}_n${END_IDX}"
    mkdir -p "$out_dir"
    out_json="${out_dir}/n${NUM_SAMPLES}_${model}_t${TEMP}_k${TOPK}_p0.95_mt${MT}${tag_suffix}.json"
    # Tagged / nonzero-seed outputs are a second pass. Leave an existing
    # file in place and go on to the next dataset.
    if [[ -n "$tag_suffix" && -e "$out_json" ]]; then
      echo "[confdyn] output already exists, leaving it and continuing: $out_json" >&2
      continue
    fi

    echo "============================================================"
    echo "[confdyn] model=$model family=$fam dataset=$ds N=$NUM_SAMPLES seed=$RANDOM_SEED"
    echo "[confdyn] GPUs=$_NGPU tp=$TP_SIZE dp=$DP_SIZE  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "[confdyn] out=$out_json"
    echo "============================================================"

    extra_args=()
    if [[ -n "${LINEAR_ATTN_BACKEND:-}" ]]; then
      extra_args+=(--linear_attn_backend "$LINEAR_ATTN_BACKEND")
    fi
    if [[ -n "${LINEAR_ATTN_DECODE_BACKEND:-}" ]]; then
      extra_args+=(--linear_attn_decode_backend "$LINEAR_ATTN_DECODE_BACKEND")
    fi
    if [[ -n "${LINEAR_ATTN_PREFILL_BACKEND:-}" ]]; then
      extra_args+=(--linear_attn_prefill_backend "$LINEAR_ATTN_PREFILL_BACKEND")
    fi
    if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
      extra_args+=(--disable_cuda_graph)
    fi
    if [[ -n "${CUDA_GRAPH_MAX_BS:-}" ]]; then
      extra_args+=(--cuda_graph_max_bs "$CUDA_GRAPH_MAX_BS")
    fi
    if [[ "${TREE_SEARCH:-0}" == "1" ]]; then
      extra_args+=(
        --tree_search
        --tree_beam "${TREE_BEAM:-2}"
        --tree_branch "${TREE_BRANCH:-3}"
        --tree_init_branch "${TREE_INIT_BRANCH:-0}"
        --tree_chunk "${TREE_CHUNK:-500}"
        --tree_warmup "${TREE_WARMUP:-0}"
        --tree_rank "${TREE_RANK:-chunk}"
        --tree_tail "${TREE_TAIL:-2048}"
        --tree_score "${TREE_SCORE:-dc}"
        --tree_keep "${TREE_KEEP:-score}"
        --save_completions
      )
      if [[ "${TREE_LAST_ALL:-0}" == "1" ]]; then
        extra_args+=(--tree_last_all)
      fi
    fi
    if [[ "${KEEP_TRUNCATED:-0}" == "1" ]]; then
      extra_args+=(--keep_truncated)
    fi

    if ! "$PY" -m confdyn \
      --model_path "$MODEL_PATH" \
      --family "$fam" \
      --dataset "$ds" \
      --start_idx "$START_IDX" \
      --end_idx "$END_IDX" \
      --num_samples "$NUM_SAMPLES" \
      --filter_k "$FILTER_K" \
      --temperature "$TEMP" \
      --top_p 0.95 \
      --top_k "$TOPK" \
      --presence_penalty "$PRESENCE" \
      --repetition_penalty 1.0 \
      --max_new_tokens "$MT" \
      --random_seed "$RANDOM_SEED" \
      --tp_size "$TP_SIZE" \
      --dp_size "$DP_SIZE" \
      --max_batch "$MAX_BATCH" \
      --top_logprobs_num "$TOP_LOGPROBS_NUM" \
      --dump_json "$out_json" \
      "${rho_args[@]}" \
      "${extra_args[@]}"
    then
      echo "[confdyn] dataset=$ds failed, continuing with the next dataset" >&2
      any_fail=1
      continue
    fi

    if ! "$PY" - <<PY
import json
path = ${out_json@Q}
with open(path) as f:
    payload = json.load(f)
st = payload.get("stats") or (payload.get("runs") or {}).get("stats") or {}
def pct(k):
    v = st.get(k)
    return f"{100*v:5.1f}%" if isinstance(v, (int, float)) else "  n/a"
print(
    f"BestTailDC={pct('vote_acc_best_tail_dc')}  "
    f"BestDC={pct('vote_acc_best_dc')}  "
    f"Majority={pct('vote_acc_majority')}  "
    f"TailDC-W={pct('vote_acc_tail_dc_weighted')}  "
    f"DC-W={pct('vote_acc_dc_weighted')}  "
    f"Standard={pct('vote_acc')}  "
    f"TailC-W={pct('vote_acc_tail_c_weighted')}  "
    f"TailC-F30={pct('vote_acc_tail_c_filter_30')}  "
    f"Pass1-F30={pct('pass@1_tail_c_filter_30')}  "
    f"TailC-F50={pct('vote_acc_tail_c_filter_50')}  "
    f"Pass1-F50={pct('pass@1_tail_c_filter_50')}  "
    f"Ours@50={pct('vote_acc_joint_dc_weighted_50')}  "
    f"Ours@60={pct('vote_acc_joint_dc_weighted_60')}  "
    f"Pass1-J50={pct('pass@1_joint_50')}  "
    f"Pass1-J60={pct('pass@1_joint_60')}"
)
PY
    then
      echo "[confdyn] dataset=$ds stats print failed" >&2
      any_fail=1
    fi
  done
done

if [[ "$any_fail" -ne 0 ]]; then
  echo "[confdyn] one or more datasets failed" >&2
  exit 1
fi
echo "[confdyn] done."
