#!/usr/bin/env bash
# 8 卡对比 FA3 vs TE fused attention 吞吐 / MFU。
# 用法: bash scripts/bench_attn_backends.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC="${NPROC:-8}"
export MASTER_PORT="${MASTER_PORT:-29531}"

ITERS="${ITERS:-8}"
SEQ="${SEQ:-8192}"
GBS="${GBS:-64}"
PRECISION="${PRECISION:-bf16}"
RECOMPUTE="${RECOMPUTE:-off}"

run_one() {
  local name="$1"; shift
  local log="${ROOT}/logs/bench_${name}_$(date +%Y%m%d_%H%M%S).log"
  echo "======== ${name} → ${log} ========"
  bash run.sh bench -- \
    --iters "${ITERS}" --seq "${SEQ}" --gbs "${GBS}" \
    --precision "${PRECISION}" --recompute "${RECOMPUTE}" \
    --no-ep-overlap --no-packing --no-cuda-graph \
    "$@" 2>&1 | tee "${log}"
  echo "[parse] ${name}"
  rg -n "throughput|TFLOP|elapsed time per iteration" "${log}" | tail -40 || true
}

mkdir -p logs
run_one fa3 --attn-backend flash --flash-attn 3
run_one fused --attn-backend fused

echo "======== done: compare logs/bench_fa3_* vs logs/bench_fused_* ========"
