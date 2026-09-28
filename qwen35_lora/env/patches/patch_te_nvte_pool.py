#!/usr/bin/env python3
"""把 TE 的 ``libtransformer_engine.so`` 重定向到「更大 NVTETensor pool」的构建。

背景
----
- 库存 TE（约 2.16）TensorAllocator 上限约 ~20k handles
- MoE + FusedAdam 在 4 卡（DP≈1）上常需要 ~25k+
- ``transformer_engine.common`` 在 import 时通过 ``ctypes.CDLL`` 加载 .so
  → 必须在 ``import transformer_engine`` **之前** 包装 CDLL

查找顺序
--------
1. 环境变量 ``TE_PATCHED_SO``
2. 本仓库 ``vendor/te_patch/libtransformer_engine.so``
3. 历史路径 ``/workspace/Qwen3.5-35B-A3B_SFT/vendor/te_patch/...``

找不到则退回库存 TE（短跑可能 OK，长训 MoE 可能踩 pool 上限）。
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_APPLIED = False
_REAL_CDLL = ctypes.CDLL
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_patched_so() -> Path | None:
    candidates = [
        os.environ.get("TE_PATCHED_SO"),
        str(_REPO_ROOT / "vendor" / "te_patch" / "libtransformer_engine.so"),
        "/workspace/Qwen3.5-35B-A3B_SFT/vendor/te_patch/libtransformer_engine.so",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.is_file():
            return p
    return None


def apply() -> str:
    """Install CDLL redirect. Call before importing transformer_engine."""
    global _APPLIED
    if _APPLIED:
        return "already-applied"

    patched_path = _resolve_patched_so()
    if patched_path is None:
        print(
            "[patch_te_nvte_pool] No enlarged TE .so found; using stock TE pool",
            flush=True,
        )
        return "missing-so"

    patched = str(patched_path)

    def _cdll(name, *args, **kwargs):  # noqa: ANN001
        s = str(name)
        if "libtransformer_engine" in s and "vendor/te_patch" not in s:
            name = patched
        return _REAL_CDLL(name, *args, **kwargs)

    ctypes.CDLL = _cdll  # type: ignore[misc, assignment]
    _APPLIED = True
    print(
        f"[patch_te_nvte_pool] CDLL redirect → {patched} (NVTETensor pool≈65535)",
        flush=True,
    )
    return "cdll-redirect"
