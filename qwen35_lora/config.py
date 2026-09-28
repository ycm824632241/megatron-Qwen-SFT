#!/usr/bin/env python3
"""训练配方：人类可读超参 + 组装成 Megatron-Bridge ``ConfigContainer``。

怎么用
------
1. 只改本文件顶部「✏️ 改这里」区块。
2. ``bash run.sh``（LoRA）或 ``bash run.sh smoke -- --full-sft``（全量）。

文件结构
--------
- 顶部常量：路径、并行、LoRA、精度、加速开关（日常改这里）
- ``Cfg``：一次运行的快照（frozen dataclass）
- ``default_cfg`` / ``smoke_cfg`` / ``bench_cfg``：模式预设
- ``build_megatron_config``：把 Cfg 填进 Bridge（一般不用改）

与 Bridge 的分工
----------------
- 本文件：业务配方（并行度、LoRA、FP8、HybridEP…）
- Bridge ``finetune``：分布式 init、加载 ckpt、DataLoader、训练循环
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

# =============================================================================
#  ✏️  改这里（日常调参只动这一段）
# =============================================================================

# --- 路径（训练前置：Megatron ckpt + 原始 JSONL 需已就绪）---
# 仓库布局：megatron_qwen35/{qwen35_lora, models, data, Megatron-Bridge, .venv}
_PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parent
# Qwen3.8-27B = dense VL（Qwen3_5ForConditionalGeneration），不能用 35B MoE ckpt。
HF_MODEL = "/kwkj-k8s/llm_team/model/qwen3.8_27b"
CKPT = str(_PROJECT / "models" / "Qwen3.8-27B_megatron")  # HF→Megatron 转换产物
TOKENIZER = f"{CKPT}/iter_0000000/tokenizer"
RAW_DATA = str(_PROJECT / "data" / "fea_bench_glm53_lite200.jsonl")
DATA_DIR = _PROJECT / "data" / "fea_bench_glm53_lite200"  # 7198 train / 73 val
SAVE_DIR = ROOT / "checkpoints" / "lora_27b_feabench_lite200_lastturn_gbs16"
# 续训：指向含 latest_checkpointed_iteration.txt 的目录；None = 从基座重新 LoRA。
LOAD_CHECKPOINT = None
NPROC = 4  # 卡数；必须能被 TP*PP*EP 整除且 ≥ 该积

# --- 并行 / 序列 / batch ---
# 27B dense：EP=1；4 卡用 TP=2 → DP=2。GBS 须整除 DP。
TP, PP, EP = 2, 1, 1
# 32K / 16K 在 4×H200 + 27B dense LoRA + TP=2 会 OOM；8k + full recompute。
SEQ_LENGTH = 8192
MICRO_BATCH = 1
# 4 卡 TP=2 → DP=2。7198 样本 + GBS=16 → 每 epoch ≈450 step，3 epoch ≈1350 step。
GLOBAL_BATCH = 16

# --- LoRA（加大容量以更好拟合长文）---
LORA_DIM = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.08
LR = 2.0e-5  # 与 hy 一致；LoRA+ B 矩阵有效 LR = 2e-5 × 16 = 3.2e-4
LORA_PLUS_RATIO = 16.0  # B 矩阵有效 LR = LR × 此值
WEIGHT_DECAY = 0.05
CLIP_GRAD = 1.0
WARMUP_FRACTION = 0.05
SEED = 42
# last_turn：只对最后一轮 assistant 计 loss（对齐 hy；tool-calling 轨迹）
LOSS_MODE: Literal["assistant", "last_turn", "full"] = "last_turn"
# last_turn 监督在对话末尾；超长轨迹丢掉前缀、保留尾巴。
SFT_TRUNCATION_SIDE: Literal["left", "right"] = "left"

# --- 训练进度 ---
EPOCHS = 3.0  # 正式训用 epoch；smoke/bench 改用 iters
EVAL_INTERVAL = 50  # ~450 step/epoch；50 能看到曲线且少打断
EVAL_ITERS = 8  # val=73；多 iter 估 masked LM loss
SAVE_INTERVAL = 50

# --- 加速（4×H200：27B dense LoRA，无 HybridEP）---
PRECISION: Literal["bf16", "fp8", "mxfp8"] = "bf16"
FLASH_ATTN: Literal[2, 3] = 3
ATTN_BACKEND: Literal["flash", "fused"] = "fused"
HYBRIDEP = False
EP_OVERLAP = False
HYBRIDEP_SMS = 32
RECOMPUTE: Literal["full", "selective", "off"] = "full"
# dense 无专家层；selective 只重算 attention / norm / GDN。
RECOMPUTE_MODULES = ("core_attn", "layernorm", "gdn_norm_out")
EP_OVERLAP_RECOMPUTE_MODULES = (
    "moe_act",
    "layernorm",
    "gdn_norm_out",
    "shared_experts",
    "core_attn",
)
EP_OVERLAP_OFFLOAD = False
EP_OVERLAP_OFFLOAD_MODULES = ("expert_fc1", "moe_act", "core_attn", "attn_proj", "qkv_linear")
CUDA_GRAPH = False
PACKING = False
# batch dataloader + num_epochs 需要 pad_cu/pad_to_max；pad-tail 在 gpt_step 截掉。
PACKING_PAD_CU_SEQLENS = True
PACKING_PAD_TO_MAX = True
PACKING_PAD_SEQ_TO_MULT = 1
# False：gpt_step 不传 packed_seq_params（dense 8192）；多段 pack 会跨样本 attend。
PACKING_ATTN_THD = False
PACKING_WORKERS = 64
DATALOADER_WORKERS = 8

# =============================================================================
#  运行时快照
# =============================================================================


@dataclass(frozen=True)
class Cfg:
    """一次运行的完整超参快照（由 mode / CLI 生成，再交给 build_megatron_config）。"""

    name: str = "train"
    dataset_root: str = str(DATA_DIR)
    save_dir: str = str(SAVE_DIR)
    load_checkpoint: str | None = LOAD_CHECKPOINT
    seq_length: int = SEQ_LENGTH
    tp: int = TP
    pp: int = PP
    ep: int = EP
    micro_batch_size: int = MICRO_BATCH
    global_batch_size: int = GLOBAL_BATCH
    lora_dim: int = LORA_DIM
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    lr: float = LR
    loraplus_lr_ratio: float = LORA_PLUS_RATIO
    weight_decay: float = WEIGHT_DECAY
    clip_grad: float = CLIP_GRAD
    warmup_fraction: float = WARMUP_FRACTION
    seed: int = SEED
    loss_mode: Literal["assistant", "last_turn", "full"] = LOSS_MODE
    truncation_side: Literal["left", "right"] = SFT_TRUNCATION_SIDE
    num_epochs: float | None = EPOCHS
    train_iters: int | None = None
    eval_interval: int = EVAL_INTERVAL
    eval_iters: int = EVAL_ITERS
    save_interval: int = SAVE_INTERVAL
    log_interval: int = 1
    enable_packing: bool = PACKING
    packing_pad_cu_seqlens: bool = PACKING_PAD_CU_SEQLENS
    packing_pad_to_max: bool = PACKING_PAD_TO_MAX
    packing_pad_seq_to_mult: int = PACKING_PAD_SEQ_TO_MULT
    packing_attn_thd: bool = PACKING_ATTN_THD
    packing_workers: int = PACKING_WORKERS
    dataloader_workers: int = DATALOADER_WORKERS
    recompute: Literal["full", "selective", "off"] = RECOMPUTE
    recompute_modules: tuple[str, ...] = RECOMPUTE_MODULES
    hybridep_num_sms: int = HYBRIDEP_SMS
    precision: Literal["bf16", "fp8", "mxfp8"] = PRECISION
    flash_attn_version: Literal[2, 3] = FLASH_ATTN
    attn_backend: Literal["flash", "fused"] = ATTN_BACKEND
    use_cuda_graph: bool = CUDA_GRAPH
    use_hybridep: bool = HYBRIDEP
    use_ep_overlap: bool = EP_OVERLAP
    ep_overlap_recompute_modules: tuple[str, ...] = EP_OVERLAP_RECOMPUTE_MODULES
    ep_overlap_offload: bool = EP_OVERLAP_OFFLOAD
    ep_overlap_offload_modules: tuple[str, ...] = EP_OVERLAP_OFFLOAD_MODULES
    overlap_ddp: bool = False  # grad_reduce/param_gather；LoRA+EP 曾在 step2 assert
    manual_gc_interval: int = 200
    peft_mode: Literal["lora", "full"] = "lora"


def default_cfg() -> Cfg:
    """正式 3-epoch last_turn 训练（LiveCodeBench_1k）。"""
    return Cfg()


def smoke_cfg() -> Cfg:
    """短序列 + 短迭代，验证环境与导入链。"""
    return Cfg(
        name="smoke",
        dataset_root=str(ROOT / "data" / "smoke"),
        save_dir=str(ROOT / "checkpoints" / "smoke"),
        load_checkpoint=None,
        seq_length=8192,
        num_epochs=None,
        train_iters=10,
        eval_interval=10_000,
        eval_iters=0,
        save_interval=10_000,
        recompute="off",
        enable_packing=False,  # smoke 关掉 packing，避开 collate cu_seqlens 对齐问题
    )


def bench_cfg(iters: int = 8) -> Cfg:
    """固定 iters 测吞吐（不跑完整 epoch）。"""
    return Cfg(
        name="bench",
        save_dir=str(ROOT / "checkpoints" / "bench"),
        load_checkpoint=None,
        num_epochs=None,
        train_iters=iters,
        eval_interval=10_000,
        eval_iters=0,
        save_interval=10**9,  # 避免 bench 末尾写 ckpt 浪费时间
    )


def validate_world_size(c: Cfg) -> None:
    """torchrun 的 WORLD_SIZE 必须能被 TP*PP*EP 整除。"""
    import os

    world = int(os.environ.get("WORLD_SIZE", "1"))
    need = c.tp * c.pp * c.ep
    if need > world or world % need != 0:
        raise SystemExit(f"WORLD_SIZE={world} 必须能被 TP*PP*EP={need} 整除且 ≥ need")


def apply_full_sft_defaults(c: Cfg) -> Cfg:
    """全量 SFT：关 LoRA、降低 LR、换保存目录、缩短序列、强制 selective recompute。"""
    kw: dict = {
        "peft_mode": "full",
        "lr": 5.0e-6,
        "loraplus_lr_ratio": 1.0,
        "weight_decay": 0.1,
        "recompute": "selective",
        "load_checkpoint": None,
    }
    if c.name == "smoke":
        kw.update(
            seq_length=2048,
            save_dir=str(ROOT / "checkpoints" / "full_sft_smoke"),
        )
    elif c.name in ("bench", "profile"):
        kw.update(
            seq_length=4096,
            save_dir=str(ROOT / "checkpoints" / f"full_sft_{c.name}"),
        )
    else:
        kw.update(
            seq_length=4096,
            save_dir=str(ROOT / "checkpoints" / "full_sft"),
        )
    return replace(c, **kw)


def summarize(c: Cfg) -> str:
    peft = (
        "full"
        if c.peft_mode == "full"
        else f"lora={c.lora_dim}/{c.lora_alpha}+{c.loraplus_lr_ratio}"
    )
    return (
        f"{c.name}: TP{c.tp}/EP{c.ep} seq={c.seq_length} GBS={c.global_batch_size} "
        f"{peft} lr={c.lr} "
        f"prec={c.precision} attn={c.attn_backend}"
        f"{'' if c.attn_backend != 'flash' else f'/FA{c.flash_attn_version}'} "
        f"pack={c.enable_packing}"
        f"{'' if not c.enable_packing else ('/thd' if c.packing_attn_thd else '/dense')} "
        f"hybridep={c.use_hybridep} "
        f"ep_overlap={c.use_ep_overlap} recompute={c.recompute} "
        f"loss={c.loss_mode} trunc={c.truncation_side} "
        f"epochs={c.num_epochs} iters={c.train_iters} "
        f"resume={c.load_checkpoint}"
    )


# =============================================================================
#  Megatron-Bridge 组装（一般不用改）
# =============================================================================

# LoRA 挂载的 Megatron 线性层名（attention + MLP/专家）
LORA_TARGETS = [
    "linear_qkv",
    "linear_proj",
    "in_proj",
    "out_proj",
    "linear_fc1",
    "linear_fc2",
]


def build_megatron_config(c: Cfg):
    """把 ``Cfg`` 转成 Bridge ``ConfigContainer``，供 ``finetune()`` 使用。

    组装顺序（讲解用）：
    1. HF text_config → AutoBridge → Megatron model provider（此时不加载权重）
    2. 并行 / TE / MoE dispatcher / recompute
    3. PEFT LoRA（跳过 MTP）+ LoRA+，或 peft_mode=full 时 peft=None
    4. GPTSFTDatasetConfig（chat + packing）
    5. optim / schedule / precision / checkpoint / logger / env_vars
    """
    from typing import Optional

    import torch
    from torch import nn
    from transformers import AutoConfig

    from megatron.bridge import AutoBridge
    from megatron.bridge.data.builders import ChatSFTPreprocessingConfig, GPTSFTDatasetConfig
    from megatron.bridge.data.packing import PackedSequenceSpecs
    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.recipes.common import _peft_common
    from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
    from megatron.bridge.training.comm_overlap import CommOverlapConfig
    from megatron.bridge.training.config import (
        OptimizerConfigOverrideProvider,
        OptimizerConfigOverrideProviderContext,
    )
    from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
    from megatron.bridge.training.mixed_precision import (
        bf16_mixed,
        bf16_with_fp8_current_scaling_mixed,
        bf16_with_mxfp8_mixed,
    )
    from megatron.core.optimizer import ParamKey

    # ----- LoRA+：B 矩阵（linear_out）用更高 LR -----
    @dataclass
    class _LoRAPlus(OptimizerConfigOverrideProvider):
        loraplus_lr_ratio: float = 16.0

        def build_config_overrides(self, context: OptimizerConfigOverrideProviderContext):
            overrides = dict(super().build_config_overrides(context) or {})
            if self.loraplus_lr_ratio <= 1.0:
                return overrides or None
            lr = float(context.optimizer_config.lr)
            mn = float(context.optimizer_config.min_lr or 0.0)
            overrides[ParamKey(name="*.adapter.linear_out.weight")] = {
                "max_lr": lr * self.loraplus_lr_ratio,
                "min_lr": mn * self.loraplus_lr_ratio,
            }
            return overrides

    # ----- 跳过 MTP 模块上的 LoRA（保留结构以加载 ckpt，但不训 MTP adapter）-----
    @dataclass
    class _SkipMtpLoRA(LoRA):
        def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None):
            parts = [p for p in (prefix or "").split(".") if p]
            if name:
                parts.append(name)
            if "mtp" in parts:
                return module
            return super().transform(module, name=name, prefix=prefix)

    # ----- 1) 模型 provider：与 HF→Megatron 转换产物一致 -----
    # 27B 权重是 ConditionalGeneration（VL）；convert.sh 产出 language_model.* 键。
    # 若强制 text CausalLM provider，会因缺 language_model 前缀 / 结构不一致而 load 失败。
    hf_cfg = AutoConfig.from_pretrained(HF_MODEL, trust_remote_code=True)
    m = AutoBridge.from_hf_config(hf_cfg).to_megatron_provider(load_weights=False)
    # 文本 SFT：冻结视觉，只训语言侧 + LoRA
    if hasattr(m, "freeze_vision_model"):
        m.freeze_vision_model = True
    if hasattr(m, "freeze_vision_projection"):
        m.freeze_vision_projection = True
    m.tensor_model_parallel_size = c.tp
    m.pipeline_model_parallel_size = c.pp
    m.pipeline_dtype = torch.bfloat16
    m.virtual_pipeline_model_parallel_size = None
    m.context_parallel_size = 1
    m.expert_model_parallel_size = c.ep
    m.expert_tensor_parallel_size = 1
    m.sequence_parallel = c.tp > 1
    m.seq_length = c.seq_length
    m.transformer_impl = "transformer_engine"
    # Qwen3.5 = GDN + Attention 混合架构。packing/THD 时必须标 hybrid，
    # 这样 gpt_step 才会填 total_tokens → PackedSeqParams.seq_idx，避免 GDN 跨样本串状态致 NaN。
    if hasattr(m, "is_hybrid_model"):
        m.is_hybrid_model = True
    # packing 默认关 TE/FA THD attention（NaN）；GDN 仍用 packed_seq_params 按段隔离。
    m.disable_thd_attention = bool(c.enable_packing and not c.packing_attn_thd)
    if c.attn_backend == "fused":
        from megatron.core.transformer.enums import AttnBackend

        m.attention_backend = AttnBackend.fused
        m.flash_attention_version = None
    else:
        m.attention_backend = None
        m.flash_attention_version = c.flash_attn_version

    # 算子融合（本地 uv 环境无 APEX fused CUDA ext，关闭 gradient_accumulation_fusion）
    m.gradient_accumulation_fusion = False
    m.cross_entropy_loss_fusion = True
    m.cross_entropy_fusion_impl = "native"
    # Qwen3.5-VL absolute RoPE 与 TE fused RoPE 不兼容，必须关。
    m.apply_rope_fusion = False
    m.bias_activation_fusion = True
    # dense 27B 的 num_moe_experts=None；不要套 35B MoE dispatcher / grouped GEMM。
    is_moe = bool(getattr(m, "num_moe_experts", None))
    if is_moe:
        m.moe_grouped_gemm = True
        m.moe_router_fusion = True
        # Triton 3.5.1 + TE 2.8 permute kernel 会因 get_int_dtype 引用炸 JIT hash；先关融合。
        m.moe_permute_fusion = False
        m.moe_router_force_load_balancing = False
        # FP8 MoE 需要 router padding；bf16 关掉以免白烧。
        m.moe_router_padding_for_fp8 = c.precision in ("fp8", "mxfp8")
        m.moe_router_dtype = "fp32"

        # MoE token dispatcher：HybridEP(flex) 或经典 alltoall
        if c.use_hybridep:
            sms = int(c.hybridep_num_sms)
            m.moe_token_dispatcher_type = "flex"
            m.moe_flex_dispatcher_backend = "hybridep"
            m.moe_flex_dispatcher_num_sms = sms
            m.moe_hybridep_num_sms = sms
            m.moe_hybridep_num_sms_preprocessing = sms
            m.moe_hybridep_pad_uneven_dispatch_inputs = True
            m.high_priority_a2a_comm_stream = True
            # EP overlap 与 shared_expert_overlap 互斥
            m.moe_shared_expert_overlap = not c.use_ep_overlap
            m.overlap_moe_expert_parallel_comm = bool(c.use_ep_overlap)
            m.delay_wgrad_compute = False
        elif c.use_ep_overlap:
            m.moe_token_dispatcher_type = "alltoall"
            m.moe_flex_dispatcher_backend = None
            m.moe_shared_expert_overlap = False
            m.overlap_moe_expert_parallel_comm = True
            m.delay_wgrad_compute = False
            m.high_priority_a2a_comm_stream = True
        else:
            m.moe_token_dispatcher_type = "alltoall"
            m.moe_flex_dispatcher_backend = None
            m.moe_shared_expert_overlap = True
            m.overlap_moe_expert_parallel_comm = False
            m.delay_wgrad_compute = False
    elif c.use_hybridep or c.use_ep_overlap:
        raise SystemExit("27B dense 不支持 HybridEP / EP-overlap，请保持 EP=1")

    if hasattr(m, "use_transformer_engine_op_fuser"):
        m.use_transformer_engine_op_fuser = False
    if hasattr(m, "moe_mlp_glu_interleave_size"):
        m.moe_mlp_glu_interleave_size = None

    if c.use_cuda_graph:
        m.cuda_graph_impl = "transformer_engine"
        m.cuda_graph_scope = ["moe_router", "moe_preprocess"]
        m.cuda_graph_warmup_steps = 3
        m.use_te_rng_tracker = True
    else:
        m.cuda_graph_impl = "none"

    # 保留 MTP 层以兼容 ckpt；loss 系数 0 表示不训 MTP
    m.mtp_num_layers = 1
    m.mtp_loss_scaling_factor = 0.0
    m.calculate_per_token_loss = True

    # 激活重算：省显存、费算力
    if c.recompute == "full":
        m.recompute_granularity, m.recompute_method, m.recompute_num_layers = "full", "uniform", 1
        m.recompute_modules = None
    elif c.recompute == "selective":
        m.recompute_granularity = "selective"
        m.recompute_method = m.recompute_num_layers = None
        m.recompute_modules = list(c.recompute_modules)
    else:
        # 真正关掉重算：不要残留 core_attn selective（那会白烧算力）
        m.recompute_granularity = m.recompute_method = m.recompute_num_layers = None
        m.recompute_modules = None

    # EP overlap 禁止 recompute_modules 含 "moe"
    if c.use_ep_overlap:
        if c.recompute == "full":
            raise SystemExit("EP_OVERLAP 不支持 full recompute，请用 selective 或 off")
        if c.recompute == "selective":
            mods = [x for x in c.ep_overlap_recompute_modules if x != "moe"]
            if not mods:
                mods = ["core_attn"]
            m.recompute_granularity = "selective"
            m.recompute_method = m.recompute_num_layers = None
            m.recompute_modules = mods
        else:
            m.recompute_granularity = m.recompute_method = m.recompute_num_layers = None
            m.recompute_modules = None
        if hasattr(m, "ep_overlap_early_attn_memory_release"):
            m.ep_overlap_early_attn_memory_release = True
        if c.ep_overlap_offload:
            m.fine_grained_activation_offloading = True
            m.offload_modules = list(c.ep_overlap_offload_modules)
            m.activation_offload_fraction = 1.0
        else:
            m.fine_grained_activation_offloading = False
            m.offload_modules = None
    else:
        m.fine_grained_activation_offloading = False
        m.offload_modules = None

    if c.use_hybridep:
        apply_flex_dispatcher_backend(m, m.moe_flex_dispatcher_backend)

    # ----- 2) ConfigContainer：LoRA 或全量 SFT -----
    if c.peft_mode == "full":
        from megatron.bridge.recipes.common import _sft_common

        cfg = _sft_common()
        cfg.model = m
        cfg.peft = None
        cfg.optimizer_config_override_provider = None
    else:
        cfg = _peft_common()
        cfg.model = m
        cfg.peft = _SkipMtpLoRA(
            target_modules=list(LORA_TARGETS),
            dim=c.lora_dim,
            alpha=c.lora_alpha,
            dropout=c.lora_dropout,
            dropout_position="pre",
            lora_A_init_method="xavier",
            lora_B_init_method="zero",
            normalize_moe_lora=False,
            share_expert_adapters=True,
        )
        cfg.optimizer_config_override_provider = _LoRAPlus(loraplus_lr_ratio=c.loraplus_lr_ratio)
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    cfg.tokenizer.tokenizer_model = TOKENIZER
    cfg.tokenizer.use_tokenizer_vocab_size = True

    # ----- 3) 数据：chat JSONL + offline packing -----
    packing = None
    if c.enable_packing:
        packing = PackedSequenceSpecs(
            packed_sequence_size=c.seq_length,
            num_tokenizer_workers=c.packing_workers,
            pad_cu_seqlens=bool(c.packing_pad_cu_seqlens),
            pad_seq_to_mult=int(c.packing_pad_seq_to_mult),
        )
    cfg.dataset = GPTSFTDatasetConfig(
        dataset_root=c.dataset_root,
        seq_length=c.seq_length,
        preprocessing=ChatSFTPreprocessingConfig(loss_mode=c.loss_mode),
        do_validation=c.eval_iters > 0,
        do_test=False,
        enable_offline_packing=c.enable_packing,
        offline_packing_specs=packing,
        # pad_to_max+pad_cu：batch sampler / num_epochs 需要；EOS pad-tail 由 gpt_step 截断。
        dataset_kwargs=(
            {"pad_to_max_length": bool(c.packing_pad_to_max)} if c.enable_packing else None
        ),
        dataloader_type="batch",
        num_workers=c.dataloader_workers,
        memmap_workers=max(2, min(c.packing_workers, 32)),
        data_sharding=True,
        pin_memory=True,
        persistent_workers=True,
        seed=c.seed,
    )

    # ----- 4) 训练步数：epoch 或固定 iters -----
    if c.num_epochs is not None:
        cfg.train.train_iters = None
        cfg.train.num_epochs = c.num_epochs
        cfg.scheduler.lr_decay_iters = None
    else:
        assert c.train_iters is not None
        cfg.train.train_iters = c.train_iters
        cfg.train.num_epochs = None
        cfg.scheduler.lr_decay_iters = c.train_iters

    cfg.train.global_batch_size = c.global_batch_size
    cfg.train.micro_batch_size = c.micro_batch_size
    cfg.train.manual_gc = c.manual_gc_interval > 0
    cfg.train.manual_gc_interval = max(c.manual_gc_interval, 1)
    cfg.validation.eval_interval = c.eval_interval
    cfg.validation.eval_iters = c.eval_iters

    cfg.scheduler.lr_decay_style = "cosine"
    cfg.scheduler.lr_warmup_iters = cfg.scheduler.lr_warmup_samples = 0
    cfg.scheduler.lr_warmup_fraction = c.warmup_fraction
    cfg.scheduler.lr_warmup_init = 0.0
    cfg.scheduler.weight_decay_incr_style = "constant"
    cfg.scheduler.start_weight_decay = cfg.scheduler.end_weight_decay = c.weight_decay

    cfg.optimizer.lr = c.lr
    cfg.optimizer.min_lr = 0.0
    cfg.optimizer.weight_decay = c.weight_decay
    cfg.optimizer.clip_grad = c.clip_grad
    cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2, cfg.optimizer.adam_eps = 0.9, 0.999, 1e-8
    cfg.optimizer.use_precision_aware_optimizer = False
    for attr in ("main_grads_dtype", "main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        setattr(cfg.optimizer, attr, torch.float32)

    # ----- 5) 精度 -----
    if c.precision == "fp8":
        cfg.mixed_precision = bf16_with_fp8_current_scaling_mixed()
    elif c.precision == "mxfp8":
        cfg.mixed_precision = bf16_with_mxfp8_mixed()
    else:
        cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    if c.use_ep_overlap:
        # LoRA + EP 下 DDP bucket overlap 曾触发 step2 assert；只开 MoE A2A overlap。
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=c.tp > 1,
            overlap_moe_expert_parallel_comm=True,
            delay_wgrad_compute=False,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
        )
    else:
        cfg.comm_overlap = None

    cfg.ddp.overlap_grad_reduce = bool(c.overlap_ddp)
    cfg.ddp.overlap_param_gather = bool(c.overlap_ddp)
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = False  # 配合 per-token loss

    # ----- 6) checkpoint：预训练加载 vs LoRA 保存 -----
    # PEFT：始终先加载基座；若 load 有 LoRA ckpt，再恢复 adapter / optim / step。
    cfg.checkpoint.pretrained_checkpoint = CKPT
    cfg.checkpoint.save = c.save_dir
    cfg.checkpoint.load = c.load_checkpoint
    cfg.checkpoint.save_interval = c.save_interval
    cfg.checkpoint.save_optim = True
    cfg.checkpoint.load_optim = bool(c.load_checkpoint)
    cfg.checkpoint.load_rng = bool(c.load_checkpoint)
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True
    cfg.checkpoint.most_recent_k = 2 if c.peft_mode == "full" else 5

    cfg.logger.log_interval = c.log_interval
    cfg.logger.tensorboard_dir = str(Path(c.save_dir) / "tb_logs")
    cfg.logger.log_throughput = True
    cfg.logger.log_throughput_to_tensorboard = True
    cfg.logger.log_timers_to_tensorboard = True
    cfg.logger.log_memory_to_tensorboard = True
    cfg.logger.log_device_memory_used = True
    cfg.logger.log_memory_interval = 50

    fa = int(c.flash_attn_version)
    use_flash = c.attn_backend == "flash"
    # EP A2A overlap 侧流需要 MAX_CONNECTIONS>1；须与 train.py 在 CUDA 初始化前设置一致。
    max_conn = 8 if c.use_ep_overlap else 1
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": max_conn,
        "NVTE_FLASH_ATTN": 1 if use_flash else 0,
        "NVTE_FUSED_ATTN": 1,
        "NVTE_FLASH_ATTN_V2": 1 if use_flash and fa == 2 else 0,
        "NVTE_FLASH_ATTN_V3": 1 if use_flash and fa == 3 else 0,
        "TORCH_NCCL_HIGH_PRIORITY": 1,
        "NCCL_NVLS_ENABLE": 1,
        "NCCL_P2P_LEVEL": "NVL",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "TORCHINDUCTOR_CACHE_DIR": str(ROOT / ".cache" / "inductor"),
        "TRITON_CACHE_DIR": str(ROOT / ".cache" / "triton"),
    }
    if hasattr(cfg, "dist") and hasattr(cfg.dist, "distributed_timeout_minutes"):
        cfg.dist.distributed_timeout_minutes = 30
    cfg.rng.seed = c.seed
    if c.use_cuda_graph:
        cfg.rng.te_rng_tracker = True
    return cfg
