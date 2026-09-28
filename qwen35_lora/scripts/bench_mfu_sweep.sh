#!/usr/bin/env bash
# 8×H200 吞吐/MFU 扫参（当前可用路径：无 EP-overlap / 无 HybridEP）。
# EP-overlap 需要 GPTModel.build_schedule_plan；Qwen3.5-VL+LoRA 包装暂无 → 跳过。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC="${NPROC:-8}"
export MASTER_PORT="${MASTER_PORT:-29541}"

ITERS="${ITERS:-8}"
SEQ="${SEQ:-8192}"
RECOMPUTE="${RECOMPUTE:-off}"
mkdir -p logs

summarize_log() {
  local log="$1"
  python3 - "$log" <<'PY'
import re, sys, statistics
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    print(f"log={p.name} MISSING"); raise SystemExit
text = p.read_text(errors="replace")
vals = [float(x) for x in re.findall(r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)", text)]
ms = [float(x) for x in re.findall(r"elapsed time per iteration \(ms\):\s*([0-9.]+)", text)]
failed = any(k in text for k in ("CUDA out of memory", "ChildFailedError", "quantizer.cpp", "build_schedule_plan"))
print(f"log={p.name} iters={len(vals)} failed={failed}")
if vals:
    steady = vals[1:] if len(vals) > 1 else vals
    print(
        f"  TFLOP/s/GPU: last={vals[-1]:.1f} mean_steady={statistics.mean(steady):.1f} "
        f"max={max(vals):.1f} all={','.join(f'{v:.1f}' for v in vals)}"
    )
if ms:
    steady_ms = ms[1:] if len(ms) > 1 else ms
    print(f"  iter_ms: last={ms[-1]:.0f} mean_steady={statistics.mean(steady_ms):.0f}")
if failed and not vals:
    for pat in ("build_schedule_plan", "quantizer.cpp", "CUDA out of memory", "AssertionError", "ValueError"):
        i = text.rfind(pat)
        if i >= 0:
            print("  err:", text[i:i+200].replace("\n", " ")[:200])
            break
PY
}

run_one() {
  local name="$1"; shift
  local log="${ROOT}/logs/bench_${name}.log"
  echo
  echo "======== ${name} → ${log} ========"
  export MASTER_PORT=$((29541 + RANDOM % 200))
  set +e
  bash run.sh bench -- \
    --iters "${ITERS}" --seq "${SEQ}" --recompute "${RECOMPUTE}" \
    --no-ep-overlap \
    "$@" 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e
  summarize_log "${log}"
  [[ ${rc} -ne 0 ]] && echo "[warn] ${name} exit=${rc} (continue)"
}

# 1) attention backends
run_one fused_bf16_gbs32_nopack \
  --attn-backend fused --precision bf16 --gbs 32 --no-packing --no-cuda-graph
run_one fa3_bf16_gbs32_nopack \
  --attn-backend flash --flash-attn 3 --precision bf16 --gbs 32 --no-packing --no-cuda-graph

# 2) packing (+ pad_cu_seqlens via config default)
run_one fused_bf16_gbs32_pack \
  --attn-backend fused --precision bf16 --gbs 32 --packing --no-cuda-graph
run_one fused_bf16_gbs64_pack \
  --attn-backend fused --precision bf16 --gbs 64 --packing --no-cuda-graph

# 3) fp8 (no EP-overlap)
run_one fused_fp8_gbs32_nopack \
  --attn-backend fused --precision fp8 --gbs 32 --no-packing --no-cuda-graph
run_one fused_fp8_gbs32_pack \
  --attn-backend fused --precision fp8 --gbs 32 --packing --no-cuda-graph

# 4) cuda graph on best packing candidate
run_one fused_bf16_gbs64_pack_cg \
  --attn-backend fused --precision bf16 --gbs 64 --packing --cuda-graph

echo
echo "======== sweep summary ========"
for f in \
  logs/bench_fused_bf16_gbs32_nopack.log \
  logs/bench_fa3_bf16_gbs32_nopack.log \
  logs/bench_fused_bf16_gbs32_pack.log \
  logs/bench_fused_bf16_gbs64_pack.log \
  logs/bench_fused_fp8_gbs32_nopack.log \
  logs/bench_fused_fp8_gbs32_pack.log \
  logs/bench_fused_bf16_gbs64_pack_cg.log
do
  [[ -f "$f" ]] && summarize_log "$f"
done
echo "======== done ========"
