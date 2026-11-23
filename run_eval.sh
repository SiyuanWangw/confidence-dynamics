#!/usr/bin/env bash
# Batch evaluation launcher for confidence-dynamics (paper methods + baselines).
#
# Examples:
#   MODELS=Qwen3-8B DATASETS="hmmt gpqa_diamond" NUM_SAMPLES=10 bash run_eval.sh
#   MODELS=gemma-4-12B FAMILY=gemma4 DATASETS="hmmt sgpqa200" bash run_eval.sh
#
# Paper defaults are applied per family:
#   Qwen:  T=0.6 top_k=20 presence_penalty=1.5  tail=2048  joint_rho=0.6
#   Gemma: T=1.0 top_k=64 presence_penalty=0    tail=20%   joint_rho=0.5
#   Countdown on Qwen uses T=1.0 (combinatorial exploration).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SGLANG_LOG_LEVEL="${SGLANG_LOG_LEVEL:-warning}"
export DUMP_STEP_STATS="${DUMP_STEP_STATS:-1}"
export OGF_LAYER_MARGINS="${OGF_LAYER_MARGINS:-1}"
export DEEPCONF_K="${DEEPCONF_K:-20}"
export LOCAL_TAIL_WINDOW="${LOCAL_TAIL_WINDOW:-2048}"

PY="${CONFDYN_PYTHON:-${OGF_PYTHON:-python}}"
MODELS_DIR="${MODELS_DIR:-${ROOT}/../models}"
MODELS="${MODELS:-Qwen3-8B}"
FAMILY="${FAMILY:-}"
DATASETS="${DATASETS:-hmmt}"
OUT_TAG="${OUT_TAG:-}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-1000}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
FILTER_K="${FILTER_K:-5}"
TOP_LOGPROBS_NUM="${TOP_LOGPROBS_NUM:-0}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
JOINT_RHO="${JOINT_RHO:-}"

_cvd="${CUDA_VISIBLE_DEVICES:-}"
_cvd="${_cvd// /}"
if [[ -z "$_cvd" ]]; then _NGPU=1
else _commas="${_cvd//[^,]}"; _NGPU=$((${#_commas} + 1)); fi
if [[ -z "${DP_SIZE}" || "${DP_SIZE}" == "0" ]]; then
  DP_SIZE=$((_NGPU / TP_SIZE))
  [[ "$DP_SIZE" -lt 1 ]] && DP_SIZE=1
fi

detect_family() {
  local name="$1"
  local blob
  blob="$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr '_' '.')"
  if [[ "$blob" == *gemma-4* || "$blob" == *gemma4* ]]; then echo gemma4
  elif [[ "$blob" == *qwen3.8* || "$blob" == *qwen3-8* ]]; then echo qwen3.8
  elif [[ "$blob" == *qwen3.5* ]]; then echo qwen3.5
  elif [[ "$blob" == *qwen3* ]]; then echo qwen3
  else echo generic
  fi
}

max_tokens_for() {
  local ds="$1"
  case "$ds" in
    hmmt|aime|aime26) echo 40000 ;;
    *) echo 32768 ;;
  esac
}

sampling_for() {
  # sets TEMP TOPK PRESENCE via namerefs-like globals
  local fam="$1" ds="$2"
  if [[ "$fam" == gemma4 ]]; then
    TEMP="${TEMPERATURE:-1.0}"; TOPK="${TOP_K:-64}"; PRESENCE="${PRESENCE_PENALTY:-0.0}"
  else
    TEMP="${TEMPERATURE:-0.6}"; TOPK="${TOP_K:-20}"; PRESENCE="${PRESENCE_PENALTY:-1.5}"
    if [[ "$ds" == countdown5 || "$ds" == countdown6 ]]; then
      TEMP="${TEMPERATURE:-1.0}"
    fi
  fi
}

mkdir -p "${ROOT}/results"

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
    tag_suffix=""
    [[ -n "$OUT_TAG" ]] && tag_suffix="_${OUT_TAG}"
    out_dir="${ROOT}/results/${ds}_n${END_IDX}"
    mkdir -p "$out_dir"
    out_json="${out_dir}/n${NUM_SAMPLES}_${model}_t${TEMP}_k${TOPK}_p0.95_mt${MT}${tag_suffix}.json"

    echo "============================================================"
    echo "[confdyn] model=$model family=$fam dataset=$ds N=$NUM_SAMPLES"
    echo "[confdyn] out=$out_json"
    echo "============================================================"

    "$PY" "${ROOT}/generate.py" \
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
      --tp_size "$TP_SIZE" \
      --dp_size "$DP_SIZE" \
      --top_logprobs_num "$TOP_LOGPROBS_NUM" \
      --dump_json "$out_json" \
      "${rho_args[@]}"

    "$PY" - <<PY
import json
path = ${out_json@Q}
with open(path) as f:
    st = json.load(f).get("stats") or {}
def pct(k):
    v = st.get(k)
    return f"{100*v:5.1f}%" if isinstance(v, (int, float)) else "  n/a"
print(
    f"Standard={pct('vote_acc')}  "
    f"TailC-W={pct('vote_acc_tail_c_weighted')}  "
    f"TailC-F20={pct('vote_acc_tail_c_filter_20')}  "
    f"TailC-F30={pct('vote_acc_tail_c_filter_30')}  "
    f"TailC-F50={pct('vote_acc_tail_c_filter_50')}  "
    f"Ours={pct('vote_acc_joint_dc_weighted')}  "
    f"retained={st.get('mean_joint_retained')}"
)
PY
  done
done

echo "[confdyn] done."
