#!/usr/bin/env python3
"""LoRA + EP A2A overlap：冻结 embedding 输出也要能反传。

问题
----
``overlap_moe_expert_parallel_comm`` 使用 fine-grained 1F1B schedule，
总会调用 ``pre_process.backward(b_grad)``。PEFT LoRA 下 embedding 冻结，
``decoder_input`` 是 ``requires_grad=False`` 的 leaf →::

  RuntimeError: element 0 of tensors does not require grad ...

修复
----
对 embedding 输出 ``detach().requires_grad_(True)``：权重仍冻结，但后续
LoRA 层能收到激活梯度，EP-overlap schedule 也能跑。
"""

from __future__ import annotations


def apply() -> str:
    from megatron.core.models.common.utils import PreProcessNode

    if getattr(PreProcessNode.forward_impl, "_qwen35_peft_grad", False):
        return "PreProcess PEFT grad already applied"

    _orig = PreProcessNode.forward_impl

    def forward_impl_peft(self):
        decoder_input = _orig(self)
        if decoder_input is not None and not decoder_input.requires_grad:
            decoder_input = decoder_input.detach().requires_grad_(True)
            self.chunk_state.decoder_input = decoder_input
            self.output = decoder_input
        return decoder_input

    forward_impl_peft._qwen35_peft_grad = True  # type: ignore[attr-defined]
    PreProcessNode.forward_impl = forward_impl_peft
    print(
        "[patch_peft_ep_overlap] PreProcessNode forces requires_grad on frozen "
        "embedding output (LoRA + EP-overlap safe)"
    )
    return "PreProcess PEFT grad applied"


if __name__ == "__main__":
    print(apply())
