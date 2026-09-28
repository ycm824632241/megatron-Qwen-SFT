#!/usr/bin/env bash
# Local (non-Docker) env bootstrap with uv for qwen35_lora.
# Prefer Chinese mirrors when GitHub / download.pytorch.org are slow.
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${PROJECT}/.uv-cache}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
PYPI_MIRROR="${PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}"

mkdir -p wheels models/hf downloads .uv-cache

if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python "${PYTHON:-python3.12}"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] python=$(python -V) cuda_home=${CUDA_HOME}"

# 1) Torch (cu128). Use pre-downloaded wheel if present.
TORCH_WHL=$(ls wheels/torch-*+cu128*cp312*.whl 2>/dev/null | head -1 || true)
if [[ -n "${TORCH_WHL}" ]]; then
  echo "[setup] installing local ${TORCH_WHL}"
  uv pip install --python .venv/bin/python "${TORCH_WHL}"
else
  echo "[setup] downloading torch cu128 from Aliyun mirror..."
  uv pip install --python .venv/bin/python torch torchvision torchaudio \
    --index-url https://mirrors.aliyun.com/pytorch-wheels/cu128
fi

# 2) Core training stack from tuna / local wheels
if ls wheels/megatron_bridge-*.whl >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python wheels/megatron_bridge-*.whl wheels/megatron_core-*.whl \
    -i "${PYPI_MIRROR}"
else
  uv pip install --python .venv/bin/python 'megatron-bridge==0.5.1' 'megatron-core==0.18.2' \
    -i "${PYPI_MIRROR}"
fi

uv pip install --python .venv/bin/python --python-platform x86_64-manylinux_2_39 \
  'nvidia-resiliency-ext==0.6.0' -i "${PYPI_MIRROR}" || \
  echo "[setup] WARNING: nvidia-resiliency-ext install failed (needs manylinux_2_39)"

uv pip install --python .venv/bin/python \
  'transformers>=5.8,<=5.12.1' peft accelerate datasets einops omegaconf hydra-core \
  rich safetensors tokenizers pyarrow packaging pyyaml tqdm regex requests numpy \
  huggingface_hub modelscope flash-linear-attention absl-py nemo-run nvidia-modelopt \
  tensorboard wandb diffusers megatron-energon \
  -i "${PYPI_MIRROR}"
# transformers upper bound sometimes drifts on tuna; accept installed version if resolve fails.
# 3) Transformer Engine (needed by Megatron-Bridge LoRA + attention).
# TE 2.17+ needs newer torch headers than 2.9.1; pin 2.8.x which builds cleanly.
CUDNN_ROOT="$(python -c 'import importlib.util, pathlib; p=importlib.util.find_spec("nvidia.cudnn"); print(pathlib.Path(list(p.submodule_search_locations)[0]))' 2>/dev/null || true)"
if [[ -n "${CUDNN_ROOT}" && -f "${CUDNN_ROOT}/include/cudnn.h" ]]; then
  export CPATH="${CUDNN_ROOT}/include:${CPATH:-}"
  export CPLUS_INCLUDE_PATH="${CUDNN_ROOT}/include:${CPLUS_INCLUDE_PATH:-}"
  export CUDNN_PATH="${CUDNN_ROOT}"
  export LIBRARY_PATH="${CUDNN_ROOT}/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="${CUDNN_ROOT}/lib:${LD_LIBRARY_PATH:-}"
fi
uv pip install --python .venv/bin/python ninja pybind11 -i "${PYPI_MIRROR}" || true
uv pip install --python .venv/bin/python --no-build-isolation \
  'transformer-engine[pytorch]==2.8.0' \
  -i "${PYPI_MIRROR}" || \
  echo "[setup] WARNING: transformer-engine install failed; smoke may need bf16/no-TE path"

# Compile Megatron dataset C++ helpers against this venv's Python
if [[ -d Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets ]]; then
  echo "[setup] compiling megatron dataset helpers..."
  make -C Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets \
    PYTHON="${PROJECT}/.venv/bin/python" || \
    echo "[setup] WARNING: helpers compile failed"
fi

# 4) Megatron-Bridge source tree (convert.sh + Megatron-LM submodule)
if [[ ! -x Megatron-Bridge/scripts/conversion/convert.sh ]]; then
  if [[ -d downloads/Megatron-Bridge-src/scripts/conversion ]]; then
    ln -sfn "${PROJECT}/downloads/Megatron-Bridge-src" Megatron-Bridge
  elif [[ -f downloads/Megatron-Bridge-main.tar.gz ]]; then
    tar -xzf downloads/Megatron-Bridge-main.tar.gz -C downloads
    mv downloads/Megatron-Bridge-main Megatron-Bridge
  else
    echo "[setup] cloning Megatron-Bridge (shallow)..."
    git clone --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/NVIDIA-NeMo/Megatron-Bridge.git downloads/Megatron-Bridge-src
    ln -sfn "${PROJECT}/downloads/Megatron-Bridge-src" Megatron-Bridge
  fi
fi
if [[ -d Megatron-Bridge && ! -d Megatron-Bridge/3rdparty/Megatron-LM/megatron ]]; then
  echo "[setup] WARNING: Megatron-LM submodule missing under Megatron-Bridge/3rdparty"
fi

# 5) NVIDIA CUDA runtime libs for torch (cu128 wheels ship without bundled libs)
uv pip install --python .venv/bin/python \
  nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12 \
  nvidia-nvjitlink-cu12 nvidia-nccl-cu12 triton sympy networkx filelock jinja2 fsspec typing-extensions \
  -i "${PYPI_MIRROR}" || echo "[setup] WARNING: some nvidia-* wheels failed; retry setup"

python - <<'PY'
import torch
print('[setup] torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
try:
    import megatron.bridge, megatron.core
    print('[setup] megatron.bridge OK; megatron.core OK')
except Exception as e:
    print('[setup] megatron import issue:', e)
PY

echo "[setup] done. Activate with: source ${PROJECT}/.venv/bin/activate"
