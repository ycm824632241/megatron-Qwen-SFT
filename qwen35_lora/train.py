#!/usr/bin/env python3
"""统一训练入口（由 ``run.sh`` 里的 torchrun 拉起）。

流程（每个 rank 各跑一份）
--------------------------
1. 解析 mode / CLI（在任何 CUDA 初始化之前）
2. 按需设置 EP-overlap 相关环境变量
3. ``env.bootstrap.apply_patches()``：GB300 / TE / FA / HybridEP 补丁
4. 选择 ``Cfg``（train / smoke / bench / profile）并应用 CLI 覆盖
5. ``build_megatron_config`` → Bridge ``ConfigContainer``
6. ``finetune(config, forward_step)``：Bridge 标准 GPT/SFT 循环

本文件不实现自定义 forward；训练步使用
``megatron.bridge.training.gpt_step.forward_step``。

用法见 ``run.sh`` / README。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Qwen3.5 LoRA — train / smoke / bench / profile")
    p.add_argument("mode", nargs="?", default="train", choices=["train", "smoke", "bench", "profile"])
    p.add_argument("--iters", type=int, default=None, help="覆盖 train_iters（smoke/bench/profile）")
    p.add_argument("--gbs", type=int, default=None)
    p.add_argument("--mbs", type=int, default=None, help="micro_batch_size")
    p.add_argument("--seq", type=int, default=None)
    p.add_argument("--precision", choices=["bf16", "fp8", "mxfp8"], default=None)
    p.add_argument("--flash-attn", type=int, choices=[2, 3], default=None)
    p.add_argument(
        "--attn-backend",
        choices=["flash", "fused"],
        default=None,
        help="TE attention path: flash (FA2/FA3) or cudnn fused attention",
    )
    p.add_argument(
        "--packing-workers",
        type=int,
        default=None,
        help="offline packing tokenizer 进程数（默认 config.PACKING_WORKERS）",
    )
    p.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable offline sequence packing",
    )
    p.add_argument(
        "--packing-attn-thd",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use TE/FA THD attention with packing (default off: dense attn, GDN still packed)",
    )
    p.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable TE cuda graph (moe_router/preprocess)",
    )
    p.add_argument("--recompute", choices=["full", "selective", "off"], default=None)
    p.add_argument("--hybridep-sms", type=int, default=None)
    p.add_argument(
        "--ep-overlap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable MoE EP A2A overlap（可与 HybridEP 同开）",
    )
    p.add_argument(
        "--overlap-ddp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable DDP grad_reduce + param_gather overlap",
    )
    p.add_argument(
        "--full-sft",
        action="store_true",
        help="全量微调：peft=None，LR=5e-6，短序列，保存到 checkpoints/full_sft*",
    )
    p.add_argument("--tp", type=int, default=None)
    p.add_argument("--pp", type=int, default=None)
    p.add_argument("--ep", type=int, default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument(
        "--load-checkpoint",
        default=None,
        help="续训目录（含 latest_checkpointed_iteration.txt）；覆盖 config.LOAD_CHECKPOINT",
    )
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--profile-start", type=int, default=3)
    p.add_argument("--profile-end", type=int, default=5)
    return p.parse_args()


def _base_for_mode(mode: str, iters: int | None):
    from config import bench_cfg, default_cfg, smoke_cfg

    if mode == "smoke":
        c = smoke_cfg()
        if iters is not None:
            c = replace(c, train_iters=iters)
        return c
    if mode == "bench":
        return bench_cfg(iters or 8)
    if mode == "profile":
        end = 5
        return bench_cfg(max(iters or 6, end + 1))
    return default_cfg()


def main():
    # ------------------------------------------------------------------
    # 1) 解析参数；EP-overlap 相关 env 必须在 CUDA context 创建前设好
    #    （驱动在创建 context 时捕获 CUDA_DEVICE_MAX_CONNECTIONS）
    # ------------------------------------------------------------------
    args = parse_args()
    from config import EP_OVERLAP, SFT_TRUNCATION_SIDE

    os.environ["QWEN35_SFT_TRUNCATION_SIDE"] = SFT_TRUNCATION_SIDE

    want_ep_overlap = EP_OVERLAP if args.ep_overlap is None else args.ep_overlap
    if want_ep_overlap:
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
        os.environ["QWEN35_EP_OVERLAP"] = "1"
        # TE>=2.10 fine-grained activation offloading 需要
        os.environ["NVTE_CPU_OFFLOAD_V1"] = "1"
    else:
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        os.environ.pop("QWEN35_EP_OVERLAP", None)

    # ------------------------------------------------------------------
    # 2) 环境补丁（mamba stub / TE pool / sm103 / HybridEP / PEFT）
    # ------------------------------------------------------------------
    from env.bootstrap import apply_patches, set_flash_attn_version

    patches = apply_patches()

    import torch
    from config import (
        CKPT,
        TOKENIZER,
        apply_full_sft_defaults,
        build_megatron_config,
        summarize,
        validate_world_size,
    )
    from megatron.bridge.training.config import ProfilingConfig
    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    # ------------------------------------------------------------------
    # 3) 组 Cfg + CLI 覆盖
    # ------------------------------------------------------------------
    c = _base_for_mode(args.mode, args.iters)
    if args.full_sft:
        c = apply_full_sft_defaults(c)
    overrides = {}
    if args.tp is not None:
        overrides["tp"] = args.tp
    if args.pp is not None:
        overrides["pp"] = args.pp
    if args.ep is not None:
        overrides["ep"] = args.ep
    if args.gbs is not None:
        overrides["global_batch_size"] = args.gbs
    if args.mbs is not None:
        overrides["micro_batch_size"] = args.mbs
    if args.seq is not None:
        overrides["seq_length"] = args.seq
    if args.precision is not None:
        overrides["precision"] = args.precision
    if args.flash_attn is not None:
        overrides["flash_attn_version"] = args.flash_attn
        overrides.setdefault("attn_backend", "flash")
    if args.attn_backend is not None:
        overrides["attn_backend"] = args.attn_backend
    if args.packing_workers is not None:
        overrides["packing_workers"] = args.packing_workers
    if args.packing is not None:
        overrides["enable_packing"] = args.packing
    if args.packing_attn_thd is not None:
        overrides["packing_attn_thd"] = args.packing_attn_thd
    if args.cuda_graph is not None:
        overrides["use_cuda_graph"] = args.cuda_graph
    if args.recompute is not None:
        overrides["recompute"] = args.recompute
    if args.hybridep_sms is not None:
        overrides["hybridep_num_sms"] = args.hybridep_sms
    if args.ep_overlap is not None:
        overrides["use_ep_overlap"] = args.ep_overlap
    if args.overlap_ddp is not None:
        overrides["overlap_ddp"] = args.overlap_ddp
    if args.save_dir is not None:
        overrides["save_dir"] = args.save_dir
    if args.load_checkpoint is not None:
        overrides["load_checkpoint"] = args.load_checkpoint
    if args.dataset_root is not None:
        overrides["dataset_root"] = args.dataset_root
    if args.mode == "profile":
        overrides.setdefault("save_dir", str(ROOT / "checkpoints" / "profile"))
        overrides["name"] = "profile"
    if overrides:
        c = replace(c, **overrides)

    set_flash_attn_version(c.flash_attn_version if c.attn_backend == "flash" else 0)
    validate_world_size(c)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("[train] patches:", patches, flush=True)
        print(
            f"[train] CUDA_DEVICE_MAX_CONNECTIONS="
            f"{os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')}",
            flush=True,
        )
        print("[train]", summarize(c), flush=True)

    Path(c.save_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4) Cfg → ConfigContainer；钉死预训练 ckpt / tokenizer
    # ------------------------------------------------------------------
    cfg = build_megatron_config(c)
    cfg.checkpoint.pretrained_checkpoint = CKPT
    cfg.tokenizer.tokenizer_model = TOKENIZER

    if args.mode == "profile":
        cfg.profiling = ProfilingConfig(
            use_pytorch_profiler=True,
            use_nsys_profiler=False,
            profile_step_start=args.profile_start,
            profile_step_end=args.profile_end,
            profile_ranks=[0],
            pytorch_profiler_collect_shapes=False,
            pytorch_profiler_collect_callstack=False,
        )

    # ------------------------------------------------------------------
    # 5) Bridge 训练入口（内部转 pretrain；要求有 pretrained_checkpoint）
    # ------------------------------------------------------------------
    finetune(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
