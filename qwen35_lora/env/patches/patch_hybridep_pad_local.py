#!/usr/bin/env python3
"""HybridEP pad：用静态 ``seq_length`` 对齐，避免与 EP A2A overlap 死锁。

问题
----
库存 ``moe_hybridep_pad_uneven_dispatch_inputs`` 在 ``setup_metadata`` 里做
EP ``all_reduce(MAX)`` + ``.item()``（常从 ``torch.compile`` 的
``dispatch_preprocess`` 进入）。这与
``overlap_moe_expert_parallel_comm`` 的 fine-grained 调度冲突 → 死锁 / 超时。

仅做本地 ``% 64`` 也不够：packing/THD 下各 rank 原始 token 数可能差 1
（例如 660 vs 659），HybridEP ALLGATHER 仍会挂。

修复
----
每个 rank 统一 pad 到同一静态大小 ``config.seq_length``（本项目 8k/32k
已满足 %64==0）。无 collective、无 D2H sync，EP-overlap 安全。

生产默认可不打开 EP overlap；打开时本补丁是前置条件。
"""

from __future__ import annotations


def apply() -> str:
    import torch
    from megatron.core.transformer.moe.fused_a2a import HYBRIDEP_TOKEN_ALIGNMENT
    from megatron.core.transformer.moe.moe_utils import (
        get_align_size_for_quantization,
        get_capacity,
    )
    from megatron.core.transformer.moe.token_dispatcher import _HybridEPManager

    if getattr(_HybridEPManager.setup_metadata, "_qwen35_static_pad", False):
        return "HybridEP static-pad already applied"

    _orig = _HybridEPManager.setup_metadata

    def setup_metadata_static_pad(self, routing_map, probs):
        num_tokens = routing_map.shape[0]
        self._original_num_tokens = num_tokens

        padded_num_tokens = num_tokens
        if self.config.moe_hybridep_pad_uneven_dispatch_inputs:
            # 各 EP rank 静态同尺寸（无 all_reduce / .item()）。
            target = int(getattr(self.config, "seq_length", num_tokens) or num_tokens)
            target = max(target, num_tokens)
            target += -target % HYBRIDEP_TOKEN_ALIGNMENT
            padded_num_tokens = target
        self._padded_num_tokens = padded_num_tokens

        routing_map = routing_map.reshape(num_tokens, self.num_experts)
        probs = probs.reshape(num_tokens, self.num_experts)
        if padded_num_tokens > num_tokens:
            pad_rows = padded_num_tokens - num_tokens
            routing_map = torch.cat(
                [routing_map, routing_map.new_zeros((pad_rows, self.num_experts))], dim=0
            )
            probs = torch.cat([probs, probs.new_zeros((pad_rows, self.num_experts))], dim=0)

        self.routing_map = routing_map
        self.token_probs = probs

        if self.moe_expert_rank_capacity_factor is not None:
            pad_multiple = get_align_size_for_quantization(self.config)
            budget = int(
                padded_num_tokens * self.config.moe_router_topk * self.moe_expert_rank_capacity_factor
            )
            budget += -budget % pad_multiple
            self.num_permuted_tokens = budget

        if self.drop_and_pad:
            num_out_tokens = padded_num_tokens * self.config.moe_router_topk
            self.capacity = get_capacity(
                num_tokens=num_out_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.capacity_factor,
            )
            self.num_permuted_tokens = self.capacity * self.group.size() * self.num_local_experts
            self.tokens_per_expert = torch.full(
                (self.num_local_experts,), self.capacity * self.group.size(), dtype=torch.long
            )

    setup_metadata_static_pad._qwen35_static_pad = True  # type: ignore[attr-defined]
    setup_metadata_static_pad._qwen35_orig = _orig  # type: ignore[attr-defined]
    _HybridEPManager.setup_metadata = setup_metadata_static_pad
    print(
        "[patch_hybridep_pad_local] HybridEP pad uses static seq_length "
        "(no EP all_reduce/.item(); EP-overlap safe)"
    )
    return "HybridEP static-pad applied"


if __name__ == "__main__":
    print(apply())
