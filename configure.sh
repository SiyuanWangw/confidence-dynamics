#!/usr/bin/env bash
# Install editable local SGLang + runtime deps into the active Python env.
# Prefer: conda activate confdyn   (or set CONFDYN_PYTHON=/path/to/python)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

PY="${CONFDYN_PYTHON:-${OGF_PYTHON:-}}"
if [[ -z "$PY" ]]; then
  if command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
  else
    echo "No python on PATH. Activate an env or set CONFDYN_PYTHON." >&2
    exit 1
  fi
fi
if [[ ! -x "$PY" ]]; then
  echo "python not executable: $PY" >&2
  exit 1
fi

PREFIX="$("$PY" -c 'import sys; print(sys.prefix)')"
PIP="$PREFIX/bin/pip"
UV="$PREFIX/bin/uv"
if [[ ! -x "$PIP" ]]; then
  echo "pip missing under $PREFIX" >&2
  exit 1
fi

if [[ ! -f "${ROOT}/sglang_pkg/python/pyproject.toml" ]]; then
  echo "Missing ${ROOT}/sglang_pkg/python/pyproject.toml" >&2
  echo "Clone SGLang v0.5.19 into sglang_pkg/ or restore the vendored tree." >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export SGLANG_BUILD_RUST_EXTS="${SGLANG_BUILD_RUST_EXTS:-none}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${SGLANG_VERSION:-0.5.19}"

echo "Installing into:"
echo "  python: $PY"
echo "  prefix: $PREFIX"
echo "  sglang: ${ROOT}/sglang_pkg/python"

"$PIP" install -U pip uv
TORCH_CUDA="${TORCH_CUDA:-cu129}"
echo "==> torch (index: https://download.pytorch.org/whl/${TORCH_CUDA})"
"$UV" pip install --python "$PY" --prerelease=allow \
  torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "==> Python runtime libs"
"$UV" pip install --python "$PY" --prerelease=allow \
  "transformers>=4.51" fastapi uvicorn uvloop orjson msgspec pydantic \
  pyzmq requests tqdm numpy einops pillow sentencepiece tiktoken \
  outlines compressed-tensors nvidia-ml-py psutil setproctitle aiohttp \
  modelscope datasets python-multipart prometheus-client soundfile \
  interegular partial_json_parser gguf scipy pybase64 xxhash zstandard \
  math_verify latex2sympy2_extended pytest ninja packaging \
  xgrammar llguidance mistral_common openai || true

echo "==> CUDA kernels (best-effort)"
set +e
"$UV" pip install --python "$PY" --prerelease=allow \
  sglang-kernel --index-url "https://docs.sglang.ai/whl/${TORCH_CUDA}/"
"$UV" pip install --python "$PY" --prerelease=allow \
  "flashinfer_python" --extra-index-url "https://docs.sglang.ai/whl/${TORCH_CUDA}/"
set -e

echo "==> editable local sglang (skip Rust extensions)"
"$PIP" install -U setuptools setuptools-scm wheel
"$PIP" uninstall -y sglang || true
"$PIP" install -e "${ROOT}/sglang_pkg/python" --no-deps --no-build-isolation

SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
PTH="${SITE}/confidence_dynamics_sglang.pth"
echo "${ROOT}/sglang_pkg/python" > "$PTH"
echo "wrote $PTH"

"$PY" - <<PY
import sys
print("executable:", sys.executable)
import sglang
print("sglang:", sglang.__file__, getattr(sglang, "__version__", "?"))
assert "sglang_pkg" in sglang.__file__ or "confidence-dynamics" in sglang.__file__, sglang.__file__
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpu:", torch.cuda.is_available())
PY

echo
echo "CPU tests:"
"$PY" -m pytest -q "${ROOT}/tests/test_aggregation.py" "${ROOT}/tests/test_filter_metrics.py"
echo "Done. Example:"
echo "  python generate.py --model_path /path/to/model --dataset hmmt --end_idx 2 --num_samples 4"
