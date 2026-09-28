# Qwen3.5 MoE 全量 SFT（Megatron-Bridge）技术说明

在现有 **Qwen3.5 / 3.6-35B-A3B + Megatron-Bridge** 管线上，把 LoRA 换成 **全参数监督微调**（full SFT）：基座所有可训练权重都更新。

LoRA 对照文档：同目录 `complete_guide_from_zero.md`。数据转换、packing、`finetune()` 循环与 LoRA 相同；差别只在 **PEFT 开关、学习率、并行/显存、checkpoint 导出**。

```text
HF → Megatron torch_dist（convert，一次性）
  → prepare_data → training/validation.jsonl
  → 改 config：peft=None + 更低 LR + 更省显存的并行
  → run.sh / train.py → finetune（offline packing + 全参）
  → 训中 validation + convert.sh export 回 HF
```

Bridge 没有现成的 `qwen35_*_sft_config`；同类 MoE 可参考 `qwen3_30b_a3b_sft_*gpu_h100_bf16_config`（`_sft_common()`）。本仓库落地方式是改 `qwen35_lora/config.py`。

本仓库已支持 ``bash run.sh … -- --full-sft``（``peft=None``、LR=$5\times 10^{-6}$、短序列）。不加该开关时 ``bash run.sh`` **仍是 LoRA**。

当前配方路径（已在 `qwen35_lora/config.py`）：

- 权重：`megatron_qwen35/models/Qwen3.6-35B-A3B_megatron_text`
- 原始数据：`megatron_qwen35/data/synthlong_raw.jsonl`（`prepare_data` 写入 `qwen35_lora/data/full`）

---

## 启动流程

工作目录：`ycm/megatron_qwen35/qwen35_lora`。卡被占满时先腾 GPU。全量至少 **4 张空闲 H200**（`TP=1,PP=1,EP=4`）；8 卡更稳（加 `--ep 8`）。

### 冒烟

```bash
cd /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora
NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh smoke -- --full-sft
```

日志应含 `full lr=5e-06`，不应再出现 `lora=64/128`。默认 seq=2048、10 iter、selective recompute。

### 正式全量

```bash
cd /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora
NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 bash run.sh -- --full-sft
```

默认 seq=4096、2 epoch、保存到 `checkpoints/full_sft/`。8 卡示例：

```bash
NPROC=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh smoke -- --full-sft --ep 8
NPROC=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run.sh -- --full-sft --ep 8
```

OOM 时加 `--seq 2048 --gbs 8 --recompute selective`（不要关 recompute）。

导出见下文第 6 步 / 第 7 节。

---

## 1. 和 LoRA 的差别

工作目录：`ycm/megatron_qwen35/qwen35_lora`。卡被占满时先腾 GPU，否则 smoke 也会失败。

```text
1. 确认基座 ckpt + 数据（LoRA 那套可复用）
2. 改 config.py：关 LoRA、降 LR、换保存目录、缩短序列
3. smoke 验能跑、确认不是 LoRA
4. bench 看显存/吞吐
5. bash run.sh 正式训
6. convert.sh export 出完整 HF
```

### 第 1 步：前置

```bash
cd /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora
ls ../models/Qwen3.6-35B-A3B_megatron_text/latest_checkpointed_iteration.txt
ls data/full/training.jsonl
```

- 缺 Megatron 权重：跑 `scripts/convert_hf_to_megatron.sh`（一次性）。
- 缺 JSONL：`run.sh` 会按 mode 调用 `prepare_data.py`；与 LoRA 共用 `data/full`，不必为全量另做一份。
- `CKPT` / `TOKENIZER` / `RAW_DATA` 仍看 `config.py` 顶部；当前副本指向 `Qwen3.6-35B-A3B_megatron_text`。

### 第 2 步：改配方（必须，否则仍是 LoRA）

只动 `config.py` 顶部常量 + `build_megatron_config` 里 PEFT 段。完整字段见下文第 3 节。最少改这些：

| 项 | 建议 |
|----|------|
| `SAVE_DIR` | `ROOT / "checkpoints" / "full_sft"`，不要写入 `checkpoints/lora` |
| `LR` | `5.0e-6`（不要再用 LoRA 的 $5\times 10^{-5}$） |
| `SEQ_LENGTH` | 先 `4096`；8 卡全量不要直接 16k |
| `TP, PP, EP` | 8 卡用 `1, 1, 8`；须与 `run.sh` 的 `NPROC`（默认 8）对齐 |
| `cfg.peft` | `None`；骨架改 `_sft_common()`；删掉 LoRA / LoRA+ |

约束：`WORLD_SIZE ≥ TP×PP×EP` 且能整除。`run.sh` 里 `NPROC="${NPROC:-8}"`。

### 第 3～5 步：启动命令（改完 config 之后）

```bash
cd /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora

bash run.sh smoke
# 日志里不应再出现 lora=64/128；应能确认 peft 未注入

bash run.sh bench -- --iters 8 --seq 4096 --precision bf16
bash run.sh
```

| 命令 | 作用 |
|------|------|
| `bash run.sh smoke` | 短序列、短 iters，验环境与导入 |
| `bash run.sh bench -- --iters 8 --seq 4096 --precision bf16` | 测显存与吞吐，不写正式 ckpt |
| `bash run.sh` | 正式训练（`EPOCHS`、按间隔存盘与 eval） |

日志：`qwen35_lora/logs/<mode>_时间戳.log`。  
Checkpoint：`qwen35_lora/checkpoints/full_sft/`。

OOM 时不要直接拉长序列：先减 `SEQ_LENGTH`、保持 selective recompute，或加 TP/PP/卡（见第 4 节）。

### 第 6 步：导出完整 HF

全量没有 adapter，**不要**用 `export_adapter.py`。

```bash
cd /kwkj-k8s/llm_team/ycm/megatron_qwen35/Megatron-Bridge
export PYTHONPATH="src:3rdparty/Megatron-LM:${PYTHONPATH:-}"

bash scripts/conversion/convert.sh export \
  --executor local \
  --device gpu \
  --hf-model /kwkj-k8s/llm_team/ycm/models/Qwen3.6-35B-A3B \
  --megatron-path /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/checkpoints/full_sft/iter_XXXXXXX \
  --hf-path /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/export/full_sft_hf \
  --tp 1 --pp 1 --ep 8
```

`--tp/--pp/--ep` 与训练保存时的拓扑对齐。推理加载该 HF 目录即可，不需要 `PeftModel`。详见第 7 节。

---

## 1. 和 LoRA 的差别

| 项 | LoRA（当前 `qwen35_lora`） | 全量 SFT |
|----|---------------------------|----------|
| `cfg.peft` | `LoRA(...)` | `None` |
| 骨架 | `_peft_common()` | `_sft_common()`（或 `_peft_common()` 后把 `peft` 置空） |
| 可训练参数 | adapter（约千万级） | 语言侧全部（视觉仍可冻） |
| 典型 LR | $10^{-5}$～$10^{-4}$，可加 LoRA+ | $5\times 10^{-6}$ 量级 |
| 优化器状态 | 只覆盖 adapter | 覆盖全部可训参数（Adam 主状态为 fp32） |
| 存盘 | adapter + 基座引用 | 完整 `torch_dist` |
| 导出 | `export_adapter.py` → HF PEFT | `convert.sh export` → 完整 HF |
| 4 卡 16k | 本机 LoRA 可跑 | 很容易 OOM，需缩短序列或加卡/加 TP·PP |

`finetune()` 不区分两种模式：只要有 `pretrained_checkpoint`（或 resume `load`），内部仍走 `pretrain`。全量 = 不注入 PEFT。

官方入口：

```python
# Llama 教程
config = llama32_1b_sft_config()          # peft=None
# 或
peft_scheme=None  # default_peft_config → None
# YAML：--peft none
```

```python
finetune(config=cfg, forward_step_func=forward_step)
```

---

## 2. 软件栈（与 LoRA 相同）

| 层 | 职责 |
|----|------|
| Megatron-Core | 并行组、MoE dispatcher、重算、分布式优化器 |
| Megatron-Bridge | `ConfigContainer`、`finetune`、`GPTSFTDatasetConfig`；`peft=None` 即全参 |
| `qwen35_lora` | 配方与启动；全量时改 `build_megatron_config`，不必另起仓库 |

环境、`PYTHONPATH`、`env/` 补丁、HF→Megatron 转换、JSONL / packing 见 LoRA 文档第 3～6 节。**同一份 `CKPT` 与 `data/full` 可直接复用。**

---

## 3. 在 `qwen35_lora` 上改全量（最小步骤）

工作目录：`ycm/megatron_qwen35/qwen35_lora`。只动 `config.py` 顶部常量 + `build_megatron_config` 里 PEFT 段。

### 3.1 顶部超参

```python
SAVE_DIR = ROOT / "checkpoints" / "full_sft"  # 不要写进 checkpoints/lora
NPROC = 8                     # 全量优先 ≥8 卡；4 卡先把 SEQ 降到 2k～4k 再 smoke
TP, PP, EP = 1, 1, 8          # 8 卡：EP=8；4 卡可试 TP=2, PP=1, EP=2 或 TP=1, PP=2, EP=2
SEQ_LENGTH = 4096             # 先短后长；16k/32k 全量通常要更多卡或 TP/PP
MICRO_BATCH = 1               # packing 仍要求 MBS=1
GLOBAL_BATCH = 32             # 须能被 DP 整除；DP = NPROC / (TP*PP*CP)

# 全量：删掉对 LoRA dim/alpha 的依赖；LR 降一档
LR = 5.0e-6
WEIGHT_DECAY = 0.1            # 官方 MoE SFT 常用 0.1；LoRA 配方里是 0.05
CLIP_GRAD = 1.0
WARMUP_FRACTION = 0.05
LOSS_MODE = "assistant"

PRECISION = "bf16"            # 先 bf16 稳住；fp8 全量另开一轮
RECOMPUTE = "selective"
RECOMPUTE_MODULES = ("moe", "moe_act", "layernorm", "gdn_norm_out")
PACKING = True                # 与 LoRA 相同；本机若 THD NaN 则 PACKING_ATTN_THD=False
```

约束不变：

$$
\mathrm{WORLD\_SIZE} \ge TP\times PP\times EP,\quad
\mathrm{WORLD\_SIZE}\bmod (TP\times PP\times EP)=0
$$

$$
DP = \frac{\mathrm{WORLD\_SIZE}}{TP\times PP\times CP},\quad
EDP = \frac{\mathrm{WORLD\_SIZE}}{ETP\times EP\times PP}
$$

`EP ≤ DP`。例如 8 卡、`TP=PP=1, EP=8, CP=1` → $DP=8, EDP=1$。

### 3.2 `build_megatron_config`：关掉 PEFT

把「`_peft_common` + LoRA + LoRA+」换成「`_sft_common` + `peft=None`」。

```python
from megatron.bridge.recipes.common import _sft_common

# 删除：_SkipMtpLoRA、_LoRAPlus、cfg.peft = ...、optimizer_config_override_provider

cfg = _sft_common()
cfg.model = m
cfg.peft = None
cfg.optimizer_config_override_provider = None
```

其余保持：HF provider、冻视觉、并行、TE/MoE、`GPTSFTDatasetConfig`、Adam、cosine、`pretrained_checkpoint=CKPT`。

视觉冻结（文本任务）仍然建议打开；这是 **冻视觉的语言全量**，不是 LoRA：

```python
if hasattr(m, "freeze_vision_model"):
    m.freeze_vision_model = True
if hasattr(m, "freeze_vision_projection"):
    m.freeze_vision_projection = True
# 不要 freeze_language_model=True，否则语言侧也不更新
```

MTP：可继续 `mtp_num_layers=1`、`mtp_loss_scaling_factor=0`，与 LoRA 一样兼容 ckpt、不计入 MTP loss。若希望 MTP 也全量训，把 scaling 改为正数并确认数据与 loss。

`sequence_parallel`：`TP>1` 时打开（官方 8 卡 30B-A3B SFT 为 `TP=4, sequence_parallel=True`）。

### 3.3 不要改的部分

- `CKPT` / `TOKENIZER`：同一份 Megatron 基座
- `prepare_data.py`、JSONL schema、`loss_mode`
- `train.py` / `run.sh`：仍调用 `finetune`
- `patch_peft_ep_overlap`：全量无 LoRA adapter 时通常不需要；EP overlap 仍按 MoE 规则处理

### 3.4 启动

```bash
cd /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora
bash run.sh smoke
# 确认日志里没有 LoRA dim，且 trainable 接近语言侧全参后再：
bash run.sh bench -- --iters 8 --seq 4096 --precision bf16
bash run.sh
```

`summarize()` 若仍打印 `lora=...`，改成打印 `peft=none lr=...`，避免误判。

---

## 4. 并行与显存

全量相对 LoRA，多出来的主要是 **全参梯度 + Adam 一阶/二阶矩**（本配方 `exp_avg` 等为 fp32）。35B 级 MoE 在 4×141GB 上很难同时撑住长序列。

### 4.1 推荐拓扑（同一模型族）

官方 **Qwen3-30B-A3B** 全量 SFT（H100，seq=2048，GBS=32）：

| 卡数 | TP | PP | EP | SP | 说明 |
|------|----|----|----|----|------|
| 8 | 4 | 2 | 4 | True | 单机 8 卡，切稠密层 + 专家 |
| 16 | 1 | 1 | 16 | False | 两机，专家切细，DP=16 |

本仓库 **35B-A3B** 可先按上表试，再按 OOM 加 PP/TP 或减 `SEQ_LENGTH`。

本机若只有 **4 卡**（当前 LoRA 默认 `TP=1,PP=1,EP=4`）：

1. `SEQ_LENGTH=2048` 或 `4096`，`RECOMPUTE=selective`（含 `moe`）
2. 试 `TP=2, PP=1, EP=2` 或 `TP=1, PP=2, EP=2`（`sequence_parallel=True` 当 `TP>1`）
3. 关 `EP_OVERLAP`、关 `CUDA_GRAPH`、先 `PACKING=False` 做 smoke
4. 仍 OOM → 加卡，不要硬堆 16k

**8×H200**：优先 `TP=1, PP=1, EP=8`，seq 从 4k 往上加；不够再 `TP=2` 或 `PP=2`。

### 4.2 Batch

$$
\mathrm{GLOBAL\_BATCH} = \mathrm{MICRO\_BATCH}\times DP\times \text{gradient\_accumulation\_steps}
$$

Offline packing 要求 `MICRO_BATCH=1`。增大吞吐优先加 pack 长度或 GBS，而不是 MBS。

### 4.3 分布式优化器

保持 `use_distributed_optimizer=True`（`_sft_common` 与现配方都已开），把优化器状态按 DP 切分。不要为了「简单」关掉它。

---

## 5. 学习率与训练长度

`_sft_common()` 默认：`max_lr=5e-6`，`adam_beta2=0.98`，`train_iters=1000`。官方 30B-A3B SFT 示例：`lr=5e-6`，`min_lr=0`，`beta2=0.95`，`weight_decay=0.1`。

建议：

- 不要沿用 LoRA 的 $5\times 10^{-5}$ 或 LoRA+（$16\times$ B 矩阵 LR）
- 先 1～2 epoch、短 seq smoke，看 train/val loss 是否发散
- `clip_grad=1.0` 保留
- 数据与 packing 缓存可与 LoRA 共用；改 `SEQ_LENGTH` 会重建 `packed/`

---

## 6. 数据与 packing

与 LoRA 文档第 5～6 节相同。

```bash
python scripts/prepare_data.py --out-dir data/full --val-ratio 0.01 --seed 42
```

`GPTSFTDatasetConfig` + `ChatSFTPreprocessingConfig(loss_mode=...)` 不用改。全量不改变 loss mask 语义。

本机若 TE/FA THD + GDN packing 会 NaN，保持 `PACKING_ATTN_THD=False`（dense attention；GDN 仍靠 `packed_seq_params` 分段）。这与是否 LoRA 无关。

---

## 7. Checkpoint 与导出

### 7.1 训练产物

`SAVE_DIR/iter_*` 为完整 Megatron `torch_dist`（权重 + 默认 `save_optim=True` 的优化器）。体积远大于 LoRA adapter。`most_recent_k` 建议调小（例如 2），避免打满盘。

Resume：把 `cfg.checkpoint.load` 指到该目录，并按需 `load_optim=True`。全量 resume 不要走 `export_adapter`。

### 7.2 导出完整 HF（不是 PEFT）

不要用 `examples/conversion/adapter/export_adapter.py`。

```bash
cd /kwkj-k8s/llm_team/ycm/megatron_qwen35/Megatron-Bridge
export PYTHONPATH="src:3rdparty/Megatron-LM:${PYTHONPATH:-}"

# 路径按本机 HF 目录与 SAVE_DIR 替换
bash scripts/conversion/convert.sh export \
  --executor local \
  --device gpu \
  --hf-model /kwkj-k8s/llm_team/ycm/models/Qwen3.6-35B-A3B \
  --megatron-path /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/checkpoints/full_sft/iter_XXXXXXX \
  --hf-path /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/export/full_sft_hf \
  --tp 1 --pp 1 --ep 8
```

`--tp/--pp/--ep` 与 **训练保存时的拓扑** 对齐更稳。导出需要完整 HF 配置（及 tokenizer）；仅 config、无 safetensors 的残缺 HF 目录不够做 roundtrip 对照，但部分 export 仍要 `config.json` / `trust_remote_code` 源。

推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "/kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/export/full_sft_hf"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True
)
```

不需要 `PeftModel`。

---

## 8. 也可用官方 recipe（不经过 `qwen35_lora`）

适合对照超参，不适合直接当 35B-A3B 配方（那是 Qwen3-30B-A3B / Llama）。

```python
from megatron.bridge.recipes.qwen.qwen3_moe import qwen3_30b_a3b_sft_config
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step

cfg = qwen3_30b_a3b_sft_config()
cfg.checkpoint.pretrained_checkpoint = "/path/to/megatron_ckpt"
# 再改 dataset 为 GPTSFTDatasetConfig(dataset_root=..., preprocessing=ChatSFT...)
finetune(config=cfg, forward_step_func=forward_step)
```

Llama YAML：`--peft none` 会换成 `llama32_1b_sft_config()`。

Qwen3.5 文本 recipe 目前只有 pretrain（`qwen35_text_35b_a3b_pretrain_*`），没有 sft 别名。

---

## 9. 建议操作顺序

可执行命令与检查项见文首 **「启动流程」**。补充：

- 最好先让同一环境的 LoRA `smoke` 通过，再改全量配置。
- 全量与 LoRA 的 val loss 不可按 step 直接对比（LR 与可训参数量不同）。
- 先看任务指标和通用能力，不要只看 train loss。

---

## 10. 常见故障

| 现象 | 处理 |
|------|------|
| 4 卡 16k OOM | 预期行为；降 seq、加 TP/PP、加卡，见 §4 |
| 仍打印 / 仍保存 LoRA | `cfg.peft` 未清空；检查是否还调用 `_peft_common` 且赋值了 LoRA |
| loss 爆炸 | LR 仍用 LoRA 档；降到 $5\times 10^{-6}$，确认 `clip_grad` |
| `WORLD_SIZE` 断言 | `NPROC` 与 `TP×PP×EP` 对齐 |
| 预训练加载失败 | `CKPT` 须为 `torch_dist` 目录；provider 须与转换时架构一致（VL `language_model.*` 前缀） |
| 误用 `export_adapter.py` | 全量没有 adapter；改用 `convert.sh export` |
| packing 首次很慢 | 与 LoRA 相同，复用 `data/full/packed` |
| 视觉也在更新 | 未设 `freeze_vision_*`；文本任务建议冻视觉 |
| 磁盘满 | 全量 ckpt 很大；减小 `most_recent_k`，`SAVE_DIR` 放到大盘 |

```bash
bash run.sh smoke
bash run.sh bench -- --iters 8 --seq 4096
bash run.sh
```
