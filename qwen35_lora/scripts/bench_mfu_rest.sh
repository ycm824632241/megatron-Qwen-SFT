#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC="${NPROC:-8}"
ITERS="${ITERS:-8}"; SEQ="${SEQ:-8192}"; RECOMPUTE="${RECOMPUTE:-off}"
mkdir -p logs
summarize_log() {
  python3 - "$1" <<'PY'
import re,sys,statistics
from pathlib import Path
p=Path(sys.argv[1]); t=p.read_text(errors='replace')
vals=[float(x) for x in re.findall(r'throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)', t)]
failed=any(k in t for k in ('ChildFailedError','CUDA out of memory','build_schedule_plan','quantizer.cpp'))
print(f'log={p.name} iters={len(vals)} failed={failed}')
if vals:
  s=vals[1:] if len(vals)>1 else vals
  print(f'  TFLOP mean_steady={statistics.mean(s):.1f} max={max(vals):.1f} all={vals}')
PY
}
run_one() {
  local name=$1; shift
  local log=$ROOT/logs/bench_${name}.log
  echo "======== $name ========"
  export MASTER_PORT=$((29550+RANDOM%100))
  set +e
  bash run.sh bench -- --iters "$ITERS" --seq "$SEQ" --recompute "$RECOMPUTE" --no-ep-overlap "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  summarize_log "$log"
  echo "[rc] $name=$rc"
}
run_one fa3_bf16_gbs32_nopack --attn-backend flash --flash-attn 3 --precision bf16 --gbs 32 --no-packing --no-cuda-graph
run_one fused_bf16_gbs32_pack --attn-backend fused --precision bf16 --gbs 32 --packing --no-cuda-graph
run_one fused_bf16_gbs64_pack --attn-backend fused --precision bf16 --gbs 64 --packing --no-cuda-graph
run_one fused_fp8_gbs32_nopack --attn-backend fused --precision fp8 --gbs 32 --no-packing --no-cuda-graph
run_one fused_fp8_gbs32_pack --attn-backend fused --precision fp8 --gbs 32 --packing --no-cuda-graph
run_one fused_bf16_gbs64_pack_cg --attn-backend fused --precision bf16 --gbs 64 --packing --cuda-graph
echo '======== FINAL ========'
for f in logs/bench_fused_bf16_gbs32_nopack.log logs/bench_fa3_bf16_gbs32_nopack.log logs/bench_fused_bf16_gbs32_pack.log logs/bench_fused_bf16_gbs64_pack.log logs/bench_fused_fp8_gbs32_nopack.log logs/bench_fused_fp8_gbs32_pack.log logs/bench_fused_bf16_gbs64_pack_cg.log; do
  [[ -f $f ]] && summarize_log "$f"
done
