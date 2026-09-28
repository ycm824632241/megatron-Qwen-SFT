#!/usr/bin/env bash
# =============================================================================
# 一键启动：数据准备（如需）+ 4 卡 torchrun
# =============================================================================
#
# 用法:
#   bash run.sh              # 正式训练（27B dense LoRA：FEA-Bench glm53_lite200 TP=2 EP=1 GBS=16 last_turn 3 epoch）
#   bash run.sh smoke        # 8k 短冒烟（验环境）
#   bash run.sh bench        # 8-iter 测吞吐
#   bash run.sh profile      # torch.profiler
#   bash run.sh bench -- --iters 8 --attn-backend fused --gbs 16 --no-packing
#   bash run.sh smoke -- --full-sft          # 全量冒烟（seq=2048）
#   bash run.sh -- --full-sft                # 全量正式训（seq=4096, LR=5e-6）
#   bash scripts/bench_mfu_sweep.sh   # 多配置对比
#
# 前置:
#   1. 本地 uv 环境 + 本仓 env 补丁（train.py 自动加载）
#   2. Megatron 格式权重（见 scripts/convert_hf_to_megatron.sh）
#   3. fea_bench_glm53_lite200（training.jsonl / validation.jsonl；7198/73，bash tools）
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PROJECT="$(cd "${ROOT}/.." && pwd)"
MB="${MEGATRON_BRIDGE:-${PROJECT}/Megatron-Bridge}"

# Prefer project uv venv when present
if [[ -x "${PROJECT}/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT}/.venv/bin/activate"
fi

# Bridge + Megatron-LM 源码优先于 site-packages
export PYTHONPATH="${MB}/src:${MB}/3rdparty/Megatron-LM:${PYTHONPATH:-}"
# 本机 0-3 被 vLLM 占用；4 卡训练走 4-7。可用环境变量覆盖。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29521}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
# TE 2.8 + torch 2.9 dynamo 对部分 TE/pybind 符号不兼容；smoke/本地默认关掉
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

MODE="${1:-train}"
shift || true
# Allow `bash run.sh smoke -- --precision bf16` style by dropping a bare `--`
if [[ "${1:-}" == "--" ]]; then
  shift || true
fi

NPROC="${NPROC:-4}"
LOG_DIR="${ROOT}/logs"
mkdir -p "${LOG_DIR}"

DATA_DIR="${PROJECT}/data/fea_bench_glm53_lite200"

prepare_data() {
  local out="$1"; shift
  if [[ -f "${out}/training.jsonl" ]]; then
    echo "[data] reuse ${out}"
    return
  fi
  echo "[data] preparing → ${out}"
  python3 "${ROOT}/scripts/prepare_data.py" --out-dir "${out}" "$@"
}

case "${MODE}" in
  train)
    if [[ ! -f "${DATA_DIR}/training.jsonl" ]]; then
      echo "error: missing ${DATA_DIR}/training.jsonl" >&2
      echo "run: python3 ${ROOT}/scripts/prepare_data.py --src ${PROJECT}/data/fea_bench_glm53_lite200.jsonl --out-dir ${DATA_DIR} --val-ratio 0.01 --seed 42" >&2
      exit 1
    fi
    echo "[data] reuse ${DATA_DIR} (fea_bench_glm53_lite200 7198/73)"
    ;;
  smoke)
    prepare_data "${ROOT}/data/smoke" \
      --max-seq-length 8192 --val-ratio 0.05 \
      --max-train 128 --max-val 8 --scan-limit 4000 --seed 42
    ;;
  bench|profile)
    prepare_data "${ROOT}/data/full_10k" --val-ratio 0.01 --seed 42 --max-train 10000
    ;;
  -h|--help|help)
    sed -n '2,16p' "$0"
    exit 0
    ;;
  *)
    echo "unknown mode: ${MODE} (train|smoke|bench|profile)"; exit 1
    ;;
esac

LOG="${LOG_DIR}/${MODE}_$(date +%Y%m%d_%H%M%S).log"
echo "[run] mode=${MODE} → ${LOG}"
torchrun --nproc_per_node="${NPROC}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  "${ROOT}/train.py" "${MODE}" "$@" \
  2>&1 | tee "${LOG}"
echo "[run] done  log=${LOG}"
