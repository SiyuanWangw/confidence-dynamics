#!/usr/bin/env bash
# Generate the N=50 pools for Section 4.2 (Table 1), then score them.
#
# One model and one dataset at a time. An existing plain N=50 dump is left
# in place. Length-capped trajectories are dropped before voting, matching
# the paper. Token-level sidecars are off; the dump already stores the
# trajectory-level confidence and dynamics used by the table.
#
#   bash scripts/run_main_table.sh
#   MODELS=Qwen3-8B DATASETS="hmmt aime" CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_main_table.sh
#   SCORE=0 bash scripts/run_main_table.sh          # generate only
#
# Models are directories under MODELS_DIR (default: ../models):
#   Qwen3-1.7B  Qwen3-8B  NVIDIA-Nemotron-3-Nano-4B-BF16
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CONFDYN_PYTHON="${CONFDYN_PYTHON:-python}"
export MODELS_DIR="${MODELS_DIR:-${ROOT}/../models}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DUMP_STEP_STATS="${DUMP_STEP_STATS:-0}"
export NUM_SAMPLES=50
export RANDOM_SEED=0
unset KEEP_TRUNCATED || true

MODELS="${MODELS:-Qwen3-1.7B Qwen3-8B NVIDIA-Nemotron-3-Nano-4B-BF16}"
DATASETS="${DATASETS:-gpqa_diamond hmmt aime sgpqa200 countdown6all}"
SCORE="${SCORE:-1}"

sampling_tag() {
  local model="$1" dataset="$2"
  case "$model" in
    *Nemotron*|*nemotron*)
      echo "t1.0_k-1"
      ;;
    *)
      if [[ "$dataset" == "countdown6all" || "$dataset" == "countdown6" ]]; then
        echo "t1.0_k20"
      else
        echo "t0.6_k20"
      fi
      ;;
  esac
}

max_tokens_tag() {
  case "$1" in
    hmmt|aime) echo 40000 ;;
    *) echo 32768 ;;
  esac
}

fail=0
for model in $MODELS; do
  for dataset in $DATASETS; do
    tag="$(sampling_tag "$model" "$dataset")"
    mt="$(max_tokens_tag "$dataset")"
    out="${ROOT}/results/${dataset}_n1000/n50_${model}_${tag}_p0.95_mt${mt}.json"
    if [[ -e "$out" ]]; then
      echo "[table1] exists, skipping generation: $out"
      continue
    fi
    # Context capping can change the max-token suffix. Skip if any plain N=50
    # dump for this sampling setting is already present.
    shopt -s nullglob
    existing=( "${ROOT}/results/${dataset}_n1000/n50_${model}_${tag}_p0.95_mt"[0-9]*.json )
    shopt -u nullglob
    plain=()
    if [[ "${#existing[@]}" -gt 0 ]]; then
      for path in "${existing[@]}"; do
        base="$(basename "$path")"
        suffix="${base#n50_${model}_${tag}_p0.95_mt}"
        suffix="${suffix%.json}"
        if [[ "$suffix" =~ ^[0-9]+$ ]]; then
          plain+=("$path")
        fi
      done
    fi
    if [[ "${#plain[@]}" -gt 0 ]]; then
      echo "[table1] exists, skipping generation: ${plain[0]}"
      continue
    fi
    echo "===== $model  $dataset  N=50  GPU ${CUDA_VISIBLE_DEVICES} ====="
    if ! MODELS="$model" DATASETS="$dataset" bash "${ROOT}/run_eval.sh"; then
      echo "[table1] generation failed: $model $dataset" >&2
      fail=1
    fi
  done
done

if [[ "$SCORE" == "1" ]]; then
  score_models=""
  for model in $MODELS; do
    case "$model" in
      Qwen3-1.7B) score_models+=" Qwen3-1.7B" ;;
      Qwen3-8B) score_models+=" Qwen3-8B" ;;
      NVIDIA-Nemotron-3-Nano-4B-BF16) score_models+=" Nemotron-3-Nano-4B" ;;
      *) echo "[table1] no Table 1 name for $model" >&2; fail=1 ;;
    esac
  done
  # shellcheck disable=SC2086
  "$CONFDYN_PYTHON" "${ROOT}/scripts/score_main_table.py" \
    --results "${ROOT}/results" \
    --models $score_models \
    --datasets $DATASETS || fail=1
fi

exit "$fail"
