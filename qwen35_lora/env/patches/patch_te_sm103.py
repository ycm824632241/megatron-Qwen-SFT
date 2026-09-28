#!/usr/bin/env python3
"""Transformer Engine attention 补丁：GB300 (sm10.3) + FlashAttention 版本锁定。

两件事
------
1. **sm10.3 → sm10.0**：库存 FA2 的 ``head_dim=256`` 门控只认到 sm10.0；
   GB300 上报 (10,3) 会被拒绝。映射后才能走 FA 路径。
2. **尊重 ``NVTE_FLASH_ATTN_V{2,3}``**：在 ``get_attention_backend`` 里
   临时关掉未选中的 FA 版本；``NVTE_FLASH_ATTN=0`` 时强制走 fused。

在 ``import transformer_engine`` 之后、真正跑 attention 之前调用即可。
"""

from __future__ import annotations


def apply() -> str:
    """Patch TE attention backend selection. Returns a short status string."""
    import os

    import transformer_engine.pytorch.attention.dot_product_attention.utils as te_utils
    from transformer_engine.pytorch.utils import get_device_compute_capability as _real_gcc

    msgs: list[str] = []
    cc = _real_gcc()
    if cc == (10, 3):

        def _mapped_gcc():
            return (10, 0)

        # get_attention_backend 通过模块 globals 绑定该函数，两边都要改。
        te_utils.get_device_compute_capability = _mapped_gcc
        import transformer_engine.pytorch.utils as _te_pytorch_utils

        _te_pytorch_utils.get_device_compute_capability = _mapped_gcc
        msgs.append("sm10.3->sm10.0")
        print("[patch_te_sm103] Mapped sm10.3 -> sm10.0 for FA head_dim<=256")

    _orig_get_attention_backend = te_utils.get_attention_backend

    def _get_attention_backend_with_version_pins(*args, **kwargs):
        fa_utils = te_utils.FlashAttentionUtils
        saved = (fa_utils.is_installed, fa_utils.v3_is_installed)
        try:
            # Force fused / non-FA path when NVTE_FLASH_ATTN=0.
            if os.getenv("NVTE_FLASH_ATTN", "1") == "0":
                fa_utils.is_installed = False
                fa_utils.v3_is_installed = False
            else:
                if os.getenv("NVTE_FLASH_ATTN_V2", "1") == "0":
                    fa_utils.is_installed = False
                if os.getenv("NVTE_FLASH_ATTN_V3", "1") == "0":
                    fa_utils.v3_is_installed = False
            return _orig_get_attention_backend(*args, **kwargs)
        finally:
            fa_utils.is_installed, fa_utils.v3_is_installed = saved

    te_utils.get_attention_backend = _get_attention_backend_with_version_pins
    msgs.append("NVTE_FLASH_ATTN_V2/V3 pins")
    print("[patch_te_sm103] Honoring NVTE_FLASH_ATTN / V2 / V3 pins")
    return "TE attention patch applied (" + ", ".join(msgs) + ")"


if __name__ == "__main__":
    print(apply())
