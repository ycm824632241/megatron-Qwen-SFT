#!/usr/bin/env python3
"""GB300 / NeMo 26.06 环境引导：环境变量 + 运行时补丁。

``train.py`` 在创建 CUDA 上下文、导入 Megatron 之前调用::

    from env.bootstrap import apply_patches, set_flash_attn_version
    apply_patches()
    set_flash_attn_version(3)  # 2/3；0=关闭 FA 走 fused

补丁顺序有依赖（尤其 TE pool 必须在 ``import transformer_engine`` 之前）。
一般不必手跑；调试可用::

    python -m env.bootstrap
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 保证 ``from env.patches...`` 在以脚本方式运行时也能找到包根。
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def apply_env_defaults() -> None:
    """设置训练常用的 CUDA / NCCL / tokenizer 环境变量（setdefault，不覆盖用户已设值）。

    EP A2A overlap 需要 ``CUDA_DEVICE_MAX_CONNECTIONS > 1``（侧流 launch）；
    由 ``train.py`` 在 import torch 前根据 ``QWEN35_EP_OVERLAP`` 决定。
    """
    max_conn = "8" if os.environ.get("QWEN35_EP_OVERLAP") == "1" else "1"
    defaults = {
        "CUDA_DEVICE_MAX_CONNECTIONS": max_conn,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NCCL_GRAPH_REGISTER": "0",
        "TOKENIZERS_PARALLELISM": "false",
        # FlashAttn 版本默认 pin 到 v3；正式开训前由 set_flash_attn_version 覆盖。
        "NVTE_FLASH_ATTN": "1",
        "NVTE_FLASH_ATTN_V2": "0",
        "NVTE_FLASH_ATTN_V3": "1",
    }
    for k, v in defaults.items():
        if k == "CUDA_DEVICE_MAX_CONNECTIONS":
            if os.environ.get("QWEN35_EP_OVERLAP") == "1":
                os.environ[k] = v
            else:
                os.environ.setdefault(k, v)
        else:
            os.environ.setdefault(k, v)


def set_flash_attn_version(version: int) -> None:
    """在 CUDA context 创建前锁定 TE 使用的 FlashAttention 大版本（2/3），或 0=关闭 FA。"""
    assert version in (0, 2, 3), version
    if version == 0:
        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_FLASH_ATTN_V2"] = "0"
        os.environ["NVTE_FLASH_ATTN_V3"] = "0"
        return
    os.environ["NVTE_FLASH_ATTN"] = "1"
    for v in (2, 3):
        os.environ[f"NVTE_FLASH_ATTN_V{v}"] = "1" if v == version else "0"


def apply_patches() -> dict[str, str]:
    """按依赖顺序应用全部运行时补丁，返回每项状态字符串（便于日志打印）。"""
    apply_env_defaults()

    from env.patches import bypass_mamba_tilelang
    from env.patches import patch_te_nvte_pool
    from env.patches import patch_te_sm103

    status: dict[str, str] = {}
    status["mamba"] = bypass_mamba_tilelang.apply()
    status["nvte_pool"] = patch_te_nvte_pool.apply()

    # sm103 补丁需要先 import TE（pool 重定向已生效）。
    import transformer_engine  # noqa: F401

    status["te_sm103"] = patch_te_sm103.apply()

    try:
        from env.patches import patch_hybridep_pad_local

        status["hybridep_pad"] = patch_hybridep_pad_local.apply()
    except Exception as exc:  # noqa: BLE001
        status["hybridep_pad"] = f"skipped: {type(exc).__name__}: {exc}"

    try:
        from env.patches import patch_peft_ep_overlap

        status["peft_ep_overlap"] = patch_peft_ep_overlap.apply()
    except Exception as exc:  # noqa: BLE001
        status["peft_ep_overlap"] = f"skipped: {type(exc).__name__}: {exc}"

    try:
        from env.patches import patch_sft_left_truncation

        status["sft_truncation"] = patch_sft_left_truncation.apply()
    except Exception as exc:  # noqa: BLE001
        status["sft_truncation"] = f"skipped: {type(exc).__name__}: {exc}"

    try:
        from env.patches import patch_packing_left_truncate

        status["packing_left_truncate"] = patch_packing_left_truncate.apply()
    except Exception as exc:  # noqa: BLE001
        status["packing_left_truncate"] = f"skipped: {type(exc).__name__}: {exc}"

    return status


if __name__ == "__main__":
    for k, v in apply_patches().items():
        print(f"{k}: {v}")
