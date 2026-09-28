#!/usr/bin/env python3
"""Offline packing：超长样本左截断，而不是直接丢掉。

问题
----
``create_hist`` 会 skip ``len(input_ids)-1 > packed_sequence_size`` 的样本。
本仓库 ``LOSS_MODE=last_turn``，监督信号在对话末尾；agent 轨迹经常远超 8k，
直接 skip 会丢掉大部分数据（这次 4831→651），2 epoch 只剩约 10 个 step。

修复
----
packing 建直方图之前，把 ``input_ids`` / ``loss_mask`` 等裁到
``packed_sequence_size+1``，**丢掉前缀、保留尾部**，让最后一轮 assistant
仍落在窗口内。随后 ``create_hist`` 不再 skip。

改完后必须删除 ``data/*/packed/`` 再开训，否则会复用旧索引。
"""

from __future__ import annotations

_KEYS = ("input_ids", "loss_mask", "context_ids", "labels")


def _left_crop_item(item: dict, stored_cap: int) -> None:
    n = len(item["input_ids"])
    if n <= stored_cap:
        return
    drop = n - stored_cap
    for key in _KEYS:
        if key not in item:
            continue
        item[key] = item[key][drop:]


def apply() -> str:
    from megatron.bridge.data.packing import algorithms
    from megatron.bridge.data.packing import offline

    if getattr(algorithms.create_hist, "_qwen35_left_truncate", False):
        return "packing left-truncate already applied"

    _orig = algorithms.create_hist

    def create_hist_left_truncate(dataset, truncate_seq_len):
        stored_cap = int(truncate_seq_len) + 1
        for item in dataset:
            _left_crop_item(item, stored_cap)
        return _orig(dataset, truncate_seq_len)

    create_hist_left_truncate._qwen35_left_truncate = True
    algorithms.create_hist = create_hist_left_truncate
    offline.create_hist = create_hist_left_truncate
    return "packing left-truncate applied (keep tail / last_turn)"
