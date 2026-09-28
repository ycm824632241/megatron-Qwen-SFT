#!/usr/bin/env bash
# Download Qwen3.5-35B-A3B-Base via hf-mirror (resumable).
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT}"

if [[ -x .venv/bin/python ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

MODEL="${HF_MODEL:-Qwen/Qwen3.5-35B-A3B-Base}"
OUT="${LOCAL_DIR:-${PROJECT}/models/hf/Qwen3.5-35B-A3B-Base}"
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
LOG="${OUT}/download.log"

mkdir -p "${OUT}"
export HF_ENDPOINT="${ENDPOINT}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "[download] model=${MODEL} -> ${OUT} endpoint=${ENDPOINT}" | tee "${LOG}"

python - <<PY 2>&1 | tee -a "${LOG}"
import os
from huggingface_hub import snapshot_download

model = os.environ.get("HF_MODEL", "${MODEL}")
out = os.environ.get("LOCAL_DIR", "${OUT}")
print(f"[download] snapshot_download {model} -> {out}")
path = snapshot_download(
    repo_id=model,
    local_dir=out,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"[download] done -> {path}")
PY

echo "[download] finished"
