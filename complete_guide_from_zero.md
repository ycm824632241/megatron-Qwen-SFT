# Qwen3.5 MoE LoRA SFT（Megatron-Bridge）技术说明

在 **4×GB300** 上，用 **Megatron-Bridge** 对 **Qwen3.5-35B-A3B** 做 packed chat **LoRA SFT**。

```text
NeMo 26.06 + env/patches
  → HF → Megatron torch_dist（convert）
  → prepare_data → training/validation.jsonl
  → run.sh / train.py → finetune（offline packing + LoRA）
  → 训中 validation + 训后 export adapter 评测
```

配置入口：`config.py` 顶部「✏️ 改这里」。启动：`bash run.sh`。

---

## 1. 软件栈

| 层 | 职责 | 本仓库接触点 |
|----|------|----------------|
| Megatron-Core | 并行组、MoE dispatcher、重算、分布式优化器、训练 step | `PYTHONPATH` → `Megatron-LM` |
| Megatron-Bridge | HF↔Megatron、`ConfigContainer`、`finetune`、`GPTSFTDatasetConfig`、PEFT | `from megatron.bridge...` |
| qwen35_lora | 配方、数据脚本、GB300 补丁、启动封装 | `config.py` / `train.py` / `env/` / `scripts/` |

```python
finetune(config=cfg, forward_step_func=forward_step)
```

`finetune` 要求 `pretrained_checkpoint`（或 resume `load`），内部走 `pretrain`：建并行模型 → 加载基座 → 注入 LoRA → DataLoader → 循环 `forward_step`。

---

## 2. 并行拓扑

### 2.1 配置项与推导量

`config.py` 显式设置模型并行与进程数：

```python
TP, PP, EP = 1, 1, 4
NPROC = 4   # torchrun --nproc_per_node
```

`CP=1`、`ETP=1`（在 `build_megatron_config` 中写死）。数据并行度由 Megatron-Core 在 `initialize_model_parallel` 中按 world size 推导：

```text
model_size = TP × PP × CP
DP  = WORLD_SIZE / model_size

expert_model_size = ETP × EP × PP
EDP = WORLD_SIZE / expert_model_size
```

本拓扑 `WORLD_SIZE=4`：

| 量 | 公式 | 值 |
|----|------|-----|
| DP | `4 / (1×1×1)` | 4 |
| EDP | `4 / (1×4×1)` | 1 |

- EP=4、EDP=1：四卡各持一部分专家，专家权重不再沿数据维复制。
- 非专家模块对应 DP=4：同构副本、不同 micro-batch，梯度在 DP 组归约。
- 约束 `EP ≤ DP`；此处 EP=DP=4。

配置面只暴露 TP/PP/EP/CP；DP/EDP 由 `WORLD_SIZE` 与上述整除关系确定。`validate_world_size`：

```text
WORLD_SIZE ≥ TP×PP×EP 且 WORLD_SIZE % (TP×PP×EP) == 0
```

变更卡数时同步修改 `NPROC`（或 `torchrun`）与 `TP/PP/EP`。

### 2.2 Batch

```python
MICRO_BATCH = 1
GLOBAL_BATCH = 16
```

```text
GLOBAL_BATCH = MICRO_BATCH × DP × gradient_accumulation_steps
```

DP=4、MBS=1 → 累积步数约 4。日志中的 step time / TFLOP/s 对应一次 **global batch** 更新。Offline packing 要求 MBS=1，吞吐调节主要依赖 pack 长度、GBS 与算子路径。

### 2.3 MoE 布局选择

Qwen3.5-35B-A3B 为稀疏 MoE（约 35B 总参 / ~3B 激活）。本规模采用 **TP=1、PP=1、EP=4**：专家维切开；非专家与本地专家分片可放单卡，避免不必要的 TP all-reduce。

---

## 3. 环境

### 3.1 镜像与 PYTHONPATH

```bash
docker pull nvcr.io/nvidia/nemo:26.06
```

```bash
export PYTHONPATH="${MEGATRON_BRIDGE:-/workspace/Megatron-Bridge}/src:\
${MEGATRON_BRIDGE:-/workspace/Megatron-Bridge}/3rdparty/Megatron-LM:${PYTHONPATH:-}"
torchrun --nproc_per_node="${NPROC:-4}" train.py ...
```

工作区 Bridge / Megatron-LM 优先于镜像 site-packages。

### 3.2 运行时补丁（`env/`，CUDA 初始化前）

| 模块 | 作用 |
|------|------|
| `bypass_mamba_tilelang` | stub `mamba_ssm`，避免 FA4 所需新 tvm-ffi 与镜像 mamba 冲突 |
| `patch_te_nvte_pool` | 可选重定向扩大 pool 的 `libtransformer_engine.so` |
| `patch_te_sm103` | sm10.3→sm10.0；TE 尊重 `NVTE_FLASH_ATTN_V*` |
| `patch_hybridep_pad_local` | HybridEP 按 `seq_length` 静态 pad（无 all_reduce/`.item()`） |
| `patch_peft_ep_overlap` | 冻结 embedding 输出 `requires_grad=True`，兼容 EP-overlap schedule |

| 变量 | 行为 |
|------|------|
| `CUDA_DEVICE_MAX_CONNECTIONS` | 默认 `1`；EP overlap 时 `8`；须在 CUDA context 创建前设定 |
| `NVTE_FLASH_ATTN_V{2,3,4}` | 三选一；由 `FLASH_ATTN` / `set_flash_attn_version` 锁定 |

---

## 4. 权重：HF → Megatron

训练读取 **`torch_dist`** 分片目录。Provider 只建结构（`load_weights=False`），权重来自 `CKPT`：

```python
text = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B-Base", trust_remote_code=True).text_config
text.architectures = ["Qwen3_5MoeForCausalLM"]
m = AutoBridge.from_hf_config(text).to_megatron_provider(load_weights=False)
...
cfg.checkpoint.pretrained_checkpoint = CKPT
cfg.checkpoint.ckpt_format = "torch_dist"
```

```bash
bash scripts/convert_hf_to_megatron.sh
# HF_MODEL=Qwen/Qwen3.5-35B-A3B-Base
# MEGATRON_PATH=/workspace/models/Qwen3.5-35B-A3B_megatron_text
```

```text
${MEGATRON_PATH}/
  latest_checkpointed_iteration.txt
  iter_0000000/
    __*_*.distcp
    tokenizer/
    metadata.json
```

```python
CKPT = "/workspace/models/Qwen3.5-35B-A3B_megatron_text"
TOKENIZER = f"{CKPT}/iter_0000000/tokenizer"
```

转换并行度可与训练不同；加载时 reshard。大 MoE 使用 GPU 分布式 import（脚本默认 `DEVICE=gpu`）。

---

## 5. 数据预处理

### 5.1 职责划分

| 阶段 | 代码 | 内容 |
|------|------|------|
| 离线切分 | `scripts/prepare_data.py` | train/val、`tools` 规范化、可选长度过滤 |
| 训练时 | `GPTSFTDatasetConfig` | chat template、`loss_mask`、offline packing |

### 5.2 JSONL schema

`dataset_root`：`training.jsonl`（必需）；`validation.jsonl`（`eval_iters>0` 时需要）。

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "content": "...", "tool_call_id": "..."},
    {"role": "assistant", "content": "最终回答"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "bash", "parameters": {...}}}
  ]
}
```

- 对话字段：`messages` / `conversation` / ShareGPT `conversations`（`from`/`value`）。
- turn 至少含 `role`、`content`；其余字段传入 `apply_chat_template`。
- `tools` 为 `list[dict]`；字符串输入由 `normalize_tools` 解析并包成 `type=function`。

### 5.3 `prepare_data.py`

1. 读取 `RAW_DATA`（或 `--src`）  
2. 过滤空 `messages`；规范化 `tools`  
3. 可选 `--max-seq-length`：模板渲染后按 token 数过滤  
4. shuffle + `val_ratio` → 写出 JSONL 与 `prepare_meta.json`  

```bash
python scripts/prepare_data.py --out-dir data/full --val-ratio 0.01 --seed 42
```

已存在 `training.jsonl` 时 `run.sh` 复用。更新原始数据后删除对应 jsonl 与 `packed/`。

### 5.4 Loss mask

```python
LOSS_MODE = "last_turn"  # assistant | last_turn | full
```

| 值 | 计入 loss 的 span |
|----|-------------------|
| `assistant` | 全部 assistant |
| `last_turn` | 仅最后一轮 assistant |
| `full` | 整段渲染序列 |

配置：`ChatSFTPreprocessingConfig(loss_mode=...)`。

---

## 6. 序列 Packing

### 6.1 定义

SFT 样本长度差异大。常规 batch 右 pad 到 `seq_length` 会产生大量无效 token。Packing 将多条样本拼入定长桶，并以 `cu_seqlens` 等元数据保持 attention 边界（THD 布局）。

### 6.2 Bridge 中的 packing 路径

| 路径 | 配置 | 适用 | `micro_batch_size` |
|------|------|------|-------------------|
| Offline packed SFT | `enable_offline_packing` + `PackedSequenceSpecs` | 文本 `GPTSFTDataset`（本仓库） | = 1 |
| In-batch packing | `enable_in_batch_packing` | Direct-HF / 部分 VLM | > 1（collate 压成单条 THD） |

本仓库使用 offline 路径：

```python
PACKING = True
PackedSequenceSpecs(
    packed_sequence_size=c.seq_length,
    num_tokenizer_workers=c.packing_workers,
    pad_seq_to_mult=1,
)
cfg.dataset.enable_offline_packing = True
```

物化产物（首次训练生成，可复用）：

```text
data/full/packed/<tokenizer指纹>/
  training_${SEQ}.idx.parquet
  validation_${SEQ}.idx.parquet
  ${SEQ}_metadata.jsonl
```

变更 `SEQ_LENGTH` 或 tokenizer 会生成新指纹目录。并行拓扑变化时按 `pad_seq_to_mult = lcm(CP 相关, TP×SP 相关)` 重建 packed 数据。

### 6.3 Offline 物化行为

对 `GPTSFTDatasetConfig` 文本 SFT/PEFT：

1. 将变长多样本离线拼成定长 packed 序列；训练侧 **MBS=1**，每条物理样本即一个 pack。  
2. 完成 tokenize、chat template、loss span 后写入 Parquet 索引；后续 epoch 只读索引，避免热路径重复模板与 bin-packing。  
3. 固定 `seq_length`、tokenizer 与算法时，pack 成员与截断边界可复现，便于对齐 loss / 吞吐实验。  

Pretrain 的文档拼接由采样器完成；SFT 需保留样本边界与 assistant mask，因此使用独立的 offline 物化 + packed runtime dataset。Direct-HF 的 in-batch packing 属于另一数据入口，本仓库 JSONL 管线不启用该开关。`PACKING=False` 时退回普通 padding。

### 6.4 与吞吐

```text
token_slots_per_step ≈ packed_sequence_size × GLOBAL_BATCH
```

增大 pack 长度可减少物理 step 与 kernel launch，同时提高激活显存。默认 `SEQ_LENGTH=32768`、`GBS=16`、`MBS=1`。

---

## 7. 配置参考（`config.py`）

日常修改顶部常量；`build_megatron_config` 写入 `ConfigContainer`。

### 7.1 路径

```python
CKPT = "/workspace/models/Qwen3.5-35B-A3B_megatron_text"
TOKENIZER = f"{CKPT}/iter_0000000/tokenizer"
RAW_DATA = "/workspace/data/<your>.jsonl"
DATA_DIR = ROOT / "data" / "full"
SAVE_DIR = ROOT / "checkpoints" / "lora"
NPROC = 4
```

### 7.2 并行 / 序列 / batch

```python
TP, PP, EP = 1, 1, 4
SEQ_LENGTH = 32768
MICRO_BATCH = 1
GLOBAL_BATCH = 16
```

| 变更 | 影响 |
|------|------|
| `SEQ_LENGTH` | 显存与 pack 缓存；需重建 `packed/` |
| `EP` / `NPROC` | 重算 DP/EDP；须满足整除 |
| `GLOBAL_BATCH` | 累积步数与有效 token/step |

### 7.3 LoRA

```python
LORA_DIM, LORA_ALPHA, LORA_DROPOUT = 32, 64, 0.08
LR = 2e-5
LORA_PLUS_RATIO = 16.0   # adapter.linear_out 的 max_lr/min_lr × 16
```

目标模块：`linear_qkv`、`linear_proj`、`in_proj`、`out_proj`、`linear_fc1`、`linear_fc2`。  
路径含 `mtp` 的模块不挂 LoRA。`mtp_num_layers=1`、`mtp_loss_scaling_factor=0`：兼容 ckpt 结构，不计入 MTP loss。

### 7.4 进度与 validation

```python
EPOCHS = 2.0
EVAL_INTERVAL = 200
EVAL_ITERS = 20
SAVE_INTERVAL = 200
```

正式训练使用 `num_epochs`；smoke/bench 使用 `train_iters`。  
`EVAL_ITERS>0` → `do_validation=True`，在 `validation.jsonl`（及 packed val）上计算 masked LM loss。  
Checkpoint：`torch_dist`，`most_recent_k=5`，目录 `SAVE_DIR`。

### 7.5 精度与通信

```python
PRECISION = "fp8"       # bf16 | fp8 | mxfp8
FLASH_ATTN = 4
HYBRIDEP = True
HYBRIDEP_SMS = 16
EP_OVERLAP = False
RECOMPUTE = "selective"
RECOMPUTE_MODULES = ("moe", "moe_act", "layernorm", "gdn_norm_out")
PACKING = True
CUDA_GRAPH = False
```

- HybridEP：`moe_token_dispatcher_type=flex`，backend `hybridep`。  
- `EP_OVERLAP=True` 时关闭 `moe_shared_expert_overlap`，且 `recompute_modules` 不得含 `moe`；长序列易 OOM。  
- 另启用：grouped GEMM、router/permute fusion、RoPE/CE fusion、distributed optimizer、grad/param overlap。  
- `CUDA_GRAPH` 与 moe recompute 不兼容。

```bash
bash run.sh bench -- --iters 8 --seq 8192 --precision bf16 --flash-attn 2
bash run.sh bench -- --iters 8 --ep-overlap --seq 8192
```

---

## 8. 训练流程

### 8.1 `run.sh`

1. 设置 `PYTHONPATH`、`CUDA_VISIBLE_DEVICES`、`MASTER_*`  
2. 按 mode 调用 `prepare_data`（无 `training.jsonl` 时）  
3. `torchrun --nproc_per_node=4 train.py <mode> ...`，日志写入 `logs/`  

| mode | 说明 |
|------|------|
| `train` | `EPOCHS`、全量数据、按间隔存盘与 eval |
| `smoke` | seq=8k、短 iters、`recompute=off`、小子集 |
| `bench` | 固定 iters，关闭 eval，测吞吐 |
| `profile` | bench + `ProfilingConfig` |

### 8.2 `train.py`

1. 解析 CLI；EP overlap 时设置 `CUDA_DEVICE_MAX_CONNECTIONS=8`  
2. `apply_patches()`  
3. 构造 `Cfg`、应用覆盖、`set_flash_attn_version`、`validate_world_size`  
4. `build_megatron_config`；写入 `CKPT` / `TOKENIZER`  
5. `finetune(config, forward_step)`  

### 8.3 `build_megatron_config` 结构

1. HF `text_config` → provider（并行、TE、FA、HybridEP、recompute、MTP）  
2. PEFT LoRA + LoRA+  
3. `GPTSFTDatasetConfig`（loss_mode、offline packing）  
4. Adam、cosine+warmup、mixed precision  
5. checkpoint、logger、`env_vars`  

### 8.4 日志

```text
[train] patches: {...}
[train] train: TP1/EP4 seq=32768 GBS=16 ... prec=fp8 FA4 hybridep=True ...
```

关注：throughput/GPU、iteration time、memory、train/validation loss。

---

## 9. 模型评测

### 9.1 训练过程中的 validation

```python
cfg.validation.eval_interval = c.eval_interval
cfg.validation.eval_iters = c.eval_iters
cfg.dataset.do_validation = c.eval_iters > 0
```

数据：`validation.jsonl`（及 packed val）。指标：与训练相同 `loss_mode` 下的 masked LM loss。  
`smoke` / `bench` 将 `eval_iters=0`，关闭 validation。

### 9.2 训后导出 LoRA（HF PEFT）

训练产物为 Megatron `torch_dist`（`SAVE_DIR/iter_*`）。导出为 HuggingFace PEFT：

```bash
cd /workspace/Megatron-Bridge
export PYTHONPATH="src:3rdparty/Megatron-LM:${PYTHONPATH:-}"

python -m torch.distributed.run --nproc_per_node=4 \
  examples/conversion/adapter/export_adapter.py \
  --hf-model-path Qwen/Qwen3.5-35B-A3B-Base \
  --lora-checkpoint /workspace/qwen35_lora/checkpoints/lora/iter_XXXXXXX \
  --output /workspace/qwen35_lora/export/lora_hf \
  --trust-remote-code \
  --dtype bf16 \
  --exclude-adapter-base-prefix mtp.layers \
  --tp 1 --pp 1 --ep 4
```

```text
export/lora_hf/
  adapter_config.json
  adapter_model.safetensors
```

推理 / 任务评测示例：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen3.5-35B-A3B-Base"
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, "/workspace/qwen35_lora/export/lora_hf")

messages = [{"role": "user", "content": "..."}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**ids, max_new_tokens=256)
print(tok.decode(out[0], skip_special_tokens=True))
```

在 `validation.jsonl` 或独立测试集上批量 generate，再接任务指标或人工/自动评判。可用 `examples/conversion/adapter/verify_adapter.py` 比对 Megatron 与 PEFT logits。

### 9.3 Megatron 侧生成

也可在保持训练并行拓扑的前提下，用 Bridge 的 generate 示例直接加载 Megatron ckpt。常见评测链路为 §9.2 导出后走 HF/PEFT。

---

## 10. 常见故障

| 现象 | 处理 |
|------|------|
| `WORLD_SIZE` 断言 | 对齐 `NPROC` 与 `TP×PP×EP` |
| OOM | 降低 `SEQ_LENGTH`；保持 selective recompute；关闭 `EP_OVERLAP` |
| 预训练加载失败 | 确认 `CKPT` 为 Megatron `torch_dist` 目录 |
| FA 版本未生效 | 在 CUDA context 前设置 `NVTE_FLASH_ATTN_V*`（经本仓 `train.py`） |
| packing 首次耗时长 | offline 物化；完成后复用 `data/*/packed` |
| HybridEP / overlap 异常 | 检查补丁；overlap 时 `recompute_modules` 不含 `moe` |
| 数据未更新 | 删除 `training.jsonl` 与 `packed/` 后重跑 |

```bash
bash run.sh smoke
bash run.sh bench -- --iters 8
bash run.sh
```
