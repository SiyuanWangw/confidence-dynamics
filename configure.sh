#!/usr/bin/env bash
# Install editable local SGLang + runtime deps into the active Python env.
# Prefer: conda activate confdyn   (or set CONFDYN_PYTHON=/path/to/python)
#
# Pins match the working `ogf` / overgen-filter stack (torch 2.13+cu129,
# transformers==5.12.1, sglang 0.5.19, …).
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
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

echo "Installing into:"
echo "  python: $PY"
echo "  prefix: $PREFIX"
echo "  sglang: ${ROOT}/sglang_pkg/python"

# CUDA graph JIT needs C++20, but CUDA 12.x nvcc rejects gcc > 13.
# Match ogf exactly: conda-forge g++ 13.3.
_CONDA_BIN="${CONDA_EXE:-}"
if [[ -z "$_CONDA_BIN" ]]; then
  if command -v conda >/dev/null 2>&1; then
    _CONDA_BIN="$(command -v conda)"
  fi
fi
if [[ -z "$_CONDA_BIN" ]]; then
  _c="$(dirname "$PY")/../../bin/conda"
  [[ -x "$_c" ]] && _CONDA_BIN="$_c"
fi
_need_gxx=1
if [[ -x "${PREFIX}/bin/x86_64-conda-linux-gnu-g++" ]]; then
  _gxx_ver="$("${PREFIX}/bin/x86_64-conda-linux-gnu-g++" -dumpversion 2>/dev/null || true)"
  [[ "$_gxx_ver" == 13.* ]] && _need_gxx=0
fi
if [[ "$_need_gxx" -eq 1 ]]; then
  if [[ -n "$_CONDA_BIN" && -x "$_CONDA_BIN" ]]; then
    echo "==> conda-forge gxx_linux-64==13.3.0 (match ogf; nvcc rejects gcc>13)"
    "$_CONDA_BIN" install -y --override-channels -c conda-forge \
      "gxx_linux-64=13.3.0" "gcc_linux-64=13.3.0" -p "$PREFIX" \
      || echo "[warn] conda gxx 13.3 install failed; set CXX to g++-13 before running"
  else
    echo "[warn] no conda binary; install gxx_linux-64=13.3.0 into this env for CUDA JIT"
  fi
fi

"$PIP" install -U pip uv
UV="$PREFIX/bin/uv"

TORCH_CUDA="${TORCH_CUDA:-cu129}"
TORCH_WHEEL="${TORCH_WHEEL:-${ROOT}/../downloads/torch-2.13.0+cu129-cp312-cp312-manylinux_2_28_x86_64.whl}"

echo "==> torch 2.13 + ${TORCH_CUDA} (pinned to match ogf)"
# Keep torch / torchvision / torchaudio on the same CUDA index.
# Later deps (kernels, etc.) must not silently upgrade torch.
_install_torch_stack() {
  if [[ -f "$TORCH_WHEEL" && "$TORCH_CUDA" == "cu129" ]]; then
    echo "using local wheel: $TORCH_WHEEL"
    "$UV" pip install --python "$PY" --prerelease=allow --reinstall "$TORCH_WHEEL"
  else
    "$UV" pip install --python "$PY" --prerelease=allow --reinstall \
      torch==2.13.0 \
      --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
  fi
  "$UV" pip install --python "$PY" --prerelease=allow --reinstall \
    torchaudio==2.11.0 torchvision==0.28.0 \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
}
_install_torch_stack
echo "==> Python runtime libs (pinned to ogf)"
set +e
"$UV" pip install --python "$PY" --prerelease=allow \
  "transformers==5.12.1" fastapi uvicorn uvloop orjson msgspec pydantic \
  pyzmq requests tqdm numpy einops pillow sentencepiece tiktoken \
  outlines==0.1.11 compressed-tensors==0.18.0 \
  nvidia-ml-py psutil setproctitle aiohttp modelscope datasets \
  python-multipart prometheus-client soundfile==0.13.1 IPython \
  interegular partial_json_parser gguf scipy pybase64 xxhash zstandard \
  math_verify latex2sympy2_extended pytest ninja packaging
"$UV" pip install --python "$PY" --prerelease=allow \
  xgrammar==0.2.1 llguidance mistral_common openai==2.6.1
set -e

echo "==> CUDA kernels (best-effort; skip if the wheel needs newer glibc)"
set +e
"$UV" pip install --python "$PY" --prerelease=allow --force-reinstall \
  sglang-kernel --index-url "https://docs.sglang.ai/whl/${TORCH_CUDA}/"
"$UV" pip install --python "$PY" --prerelease=allow --force-reinstall \
  sgl-deep-gemm --index-url "https://docs.sglang.ai/whl/${TORCH_CUDA}/" --no-deps
"$UV" pip install --python "$PY" --prerelease=allow \
  "flashinfer_python==0.6.18" --extra-index-url "https://docs.sglang.ai/whl/${TORCH_CUDA}/"
set -e

echo "==> CUTLASS CuTe DSL for FlashInfer GDN (cu12; no cuda-python 13)"
set +e
"$PIP" install --no-deps \
  "nvidia-cutlass-dsl==4.6.2" \
  "nvidia-cutlass-dsl-libs-base==4.6.2" \
  "nvidia-cutlass-dsl-libs-core==4.6.2" \
  "nvidia-cutlass-dsl-libs-cu12==4.6.2" \
  "nvidia-mathdx==25.6.0" \
  "nvidia-cuda-nvdisasm==13.3.73"
set -e

echo "==> editable local sglang (skip Rust extensions)"
"$PIP" install -U setuptools setuptools-scm wheel
"$PIP" uninstall -y sglang || true
"$PIP" install -e "${ROOT}/sglang_pkg/python" --no-deps --no-build-isolation

SITE="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
PTH="${SITE}/confidence_dynamics_sglang.pth"
echo "${ROOT}/sglang_pkg/python" > "$PTH"
echo "wrote $PTH"

echo "==> editable confdyn package"
"$PIP" install -e "${ROOT}" --no-deps

echo "==> re-pin torch stack (deps above may have upgraded torch)"
set +e
_install_torch_stack
set -e

"$PY" - <<PY
import sys
print("executable:", sys.executable)
import sglang
print("sglang:", sglang.__file__, getattr(sglang, "__version__", "?"))
assert "sglang_pkg" in sglang.__file__ or "confidence-dynamics" in sglang.__file__, sglang.__file__
import confdyn
print("confdyn:", confdyn.__file__, getattr(confdyn, "__version__", "?"))
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpu:", torch.cuda.is_available())
try:
    import transformers
    print("transformers:", transformers.__version__)
except Exception as e:
    print("transformers:", type(e).__name__, e)
try:
    import cutlass.cute as cute
    import flashinfer.gdn_prefill  # noqa: F401
    print("cutlass.cute:", cute.__file__)
    print("flashinfer.gdn_prefill: OK")
except Exception as e:
    print("FlashInfer GDN cutlass import failed:", type(e).__name__, e)
PY

echo
echo "CPU tests:"
"$PY" -m pytest -q "${ROOT}/tests/"
echo "Done. Example:"
echo "  cd ${ROOT} && python -m confdyn --model_path /path/to/model --dataset hmmt --end_idx 2 --num_samples 4"
echo "CUDA index used: TORCH_CUDA=${TORCH_CUDA}  (override with TORCH_CUDA=cu126 bash configure.sh)"
