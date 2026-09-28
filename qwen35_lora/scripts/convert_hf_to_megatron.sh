#!/usr/bin/env bash
# =============================================================================
# HF → Megatron 权重转换（训练前置，一次性）
# =============================================================================
#
# 本项目训练读取的是 Megatron ``torch_dist`` checkpoint，不是 HF safetensors。
# 转换使用 Megatron-Bridge 官方 launcher：
#   Megatron-Bridge/scripts/conversion/convert.sh
#
# 本脚本只是带注释的薄封装，方便讲解与复现。
#
# 用法示例（在 NeMo 26.06 容器内，已设置 PYTHONPATH）::
#
#   # 文本 MoE（与本仓库 config 一致：Qwen3_5MoeForCausalLM）
#   bash scripts/convert_hf_to_megatron.sh
#
#   # 自定义路径
#   HF_MODEL=Qwen/Qwen3.5-35B-A3B-Base \
#   MEGATRON_PATH=/workspace/models/Qwen3.5-35B-A3B_megatron_text \
#   bash scripts/convert_hf_to_megatron.sh
#
# 产物结构（训练时 CKPT 指向此目录）::
#
#   ${MEGATRON_PATH}/
#     latest_checkpointed_iteration.txt
#     iter_0000000/
#       __*_*.distcp          # torch_dist 分片
#       metadata.json
#       tokenizer/            # 训练 tokenizer 也用这份
#       run_config.yaml
#
# 注意
# ----
# - 转换时的 TP/EP 不必等于训练拓扑；torch_dist 加载时可 reshard。
# - 大 MoE 建议 GPU 分布式 import（见下方 DEVICE=gpu）。
# - 转换后可用 Bridge roundtrip / compare 验权重（见 Megatron-Bridge 文档）。
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="$(cd "${ROOT}/.." && pwd)"
MB="${MEGATRON_BRIDGE:-${PROJECT}/Megatron-Bridge}"

# Prefer project uv venv when present
if [[ -x "${PROJECT}/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT}/.venv/bin/activate"
fi

# --- 可覆盖参数 ---
HF_MODEL="${HF_MODEL:-${PROJECT}/models/hf/Qwen3.5-35B-A3B-Base}"
MEGATRON_PATH="${MEGATRON_PATH:-${PROJECT}/models/Qwen3.5-35B-A3B_megatron_text}"
DEVICE="${DEVICE:-gpu}"                 # cpu | gpu
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
TP="${TP:-1}"
PP="${PP:-1}"
EP="${EP:-4}"                           # 与训练默认 EP 对齐更直观，非强制
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
EXECUTOR="${EXECUTOR:-local}"

export PYTHONPATH="${MB}/src:${MB}/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

CONVERT="${MB}/scripts/conversion/convert.sh"
if [[ ! -x "${CONVERT}" ]]; then
  echo "error: convert.sh not found at ${CONVERT}" >&2
  echo "set MEGATRON_BRIDGE=/path/to/Megatron-Bridge" >&2
  exit 1
fi

echo "[convert] HF_MODEL=${HF_MODEL}"
echo "[convert] MEGATRON_PATH=${MEGATRON_PATH}"
echo "[convert] device=${DEVICE} tp=${TP} pp=${PP} ep=${EP}"

ARGS=(
  import
  --executor "${EXECUTOR}"
  --device "${DEVICE}"
  --hf-model "${HF_MODEL}"
  --megatron-path "${MEGATRON_PATH}"
  --torch-dtype "${TORCH_DTYPE}"
  --trust-remote-code
)

if [[ "${DEVICE}" == "gpu" ]]; then
  ARGS+=(--gpus-per-node "${GPUS_PER_NODE}" --tp "${TP}" --pp "${PP}" --ep "${EP}")
fi

# 在 Bridge 仓库根目录调用，保证相对路径与 NeMo-Run 配置正确。
cd "${MB}"
bash "${CONVERT}" "${ARGS[@]}"

echo "[convert] done → ${MEGATRON_PATH}"
echo "[convert] next: point config.CKPT at this path, then bash ${ROOT}/run.sh smoke"
