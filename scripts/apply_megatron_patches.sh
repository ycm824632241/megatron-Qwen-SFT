#!/usr/bin/env bash
# Optional: rebuild Megatron-Bridge from upstream instead of using the vendored
# copy. Clones Megatron-Bridge (+ Megatron-LM submodule) at the pinned commits,
# applies patches/, and points the Megatron-Bridge symlink at the clone.
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT}"

if [[ -d Megatron-Bridge && ! -L Megatron-Bridge ]]; then
  echo "[patches] vendored Megatron-Bridge/ already contains the patched source; nothing to do."
  echo "[patches] to rebuild from upstream, move Megatron-Bridge/ away and rerun."
  exit 0
fi

MB_DIR="downloads/Megatron-Bridge-src"
MB_COMMIT="$(cat patches/MEGATRON_BRIDGE_COMMIT)"
MLM_COMMIT="$(cat patches/MEGATRON_LM_COMMIT)"

mkdir -p downloads
if [[ ! -d "${MB_DIR}/.git" ]]; then
  git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git "${MB_DIR}"
fi

apply_patch() {
  local repo="$1" commit="$2" patch="$3"
  if git -C "${repo}" apply --reverse --check "${patch}" 2>/dev/null; then
    echo "[patches] already applied: ${patch##*/}"
    return
  fi
  git -C "${repo}" fetch origin "${commit}" || true
  git -C "${repo}" checkout "${commit}"
  git -C "${repo}" apply --check "${patch}"
  git -C "${repo}" apply "${patch}"
  echo "[patches] applied: ${patch##*/}"
}

cd "${MB_DIR}"
apply_patch . "${MB_COMMIT}" "${PROJECT}/patches/megatron-bridge.patch"
git submodule update --init 3rdparty/Megatron-LM
apply_patch 3rdparty/Megatron-LM "${MLM_COMMIT}" "${PROJECT}/patches/megatron-lm.patch"

cd "${PROJECT}"
ln -sfn "${PROJECT}/${MB_DIR}" Megatron-Bridge
echo "[patches] Megatron-Bridge ${MB_COMMIT} + Megatron-LM ${MLM_COMMIT} patched."
