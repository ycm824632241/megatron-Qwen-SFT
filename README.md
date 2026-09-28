# megatron_qwen

基于 [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) 在 H200 上对 Qwen 系列模型做 SFT（LoRA 与全量微调）的训练工程。已用于 Qwen3.5-35B-A3B、Qwen3.6-35B-A3B（MoE）和 Qwen3.8-27B（dense，VL 结构）。

本仓库**只包含代码、配置和文档**。模型权重、训练数据、checkpoint、导出模型、预编译 wheel 和虚拟环境都不在仓库里，需要按下面步骤自行准备。

> 训练代码目录 `qwen35_lora/` 沿用了最初的命名，实际同时支持上述各模型的 LoRA 和全量微调。

---

## 项目来源与修改内容

### 来源

- **训练框架**：[NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)，commit `292bf0f`（见 `patches/MEGATRON_BRIDGE_COMMIT`）
- **底层并行库**：Megatron-Bridge 的子模块 [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)（Megatron-Core），commit `6513e3e`（见 `patches/MEGATRON_LM_COMMIT`）
- **本仓库自研部分**：`qwen35_lora/`（训练配方、启动脚本、运行时补丁）和 `scripts/`（环境安装、数据转换），不来自上述上游仓库。训练循环本身直接调用 Bridge 的 `finetune()` 和 `gpt_step.forward_step`，没有自己实现。

### 对上游源码的修改（`patches/`）

仓库内的 `Megatron-Bridge/`（含 `3rdparty/Megatron-LM/`）就是上述两个 commit 的源码 + 下面列出的修改，可直接使用。同样的修改也以 diff 形式保存在 `patches/` 中，便于查看改动和对照上游。上游许可证保留在 `Megatron-Bridge/LICENSE`（Apache-2.0）和 `Megatron-Bridge/3rdparty/Megatron-LM/LICENSE`。

修改集中解决两类问题：Qwen3.5 系列的 **GDN（Gated DeltaNet）+ Attention 混合架构在 offline packing 下反传出现 NaN**，以及 **本地 uv 环境（无 APEX）下的兼容问题**。

**Megatron-Bridge**（9 个文件，`megatron-bridge.patch`）：

| 文件 | 修改 |
|------|------|
| `training/utils/packed_seq_utils.py` | 新增 `trim_pack_eos_pad_tail`：把 packing 尾部的 EOS 填充从 loss 中屏蔽，并去掉 `cu_seqlens` 末尾的零长度段（零长度序列会让 GDN / causal_conv1d 反传 NaN） |
| `training/gpt_step.py` | 前向前调用上面的裁剪；开启 `disable_thd_attention` 时不再传 `packed_seq_params`，改走 dense attention；FLOPs 统计同步改为按 dense 计算 |
| `data/packing/gpt_sft.py` | packing 尾部填充只记在 `cu_seqlens_padded` 里，不再算作一条真实序列 |
| `data/packing/offline.py` | packing 的 tokenizer 进程数 `-1` 时自动取 CPU 核数的一半，memmap 进程数跟随调大，加快首次 packing |
| `data/conversation_processing.py` | Qwen chat template 在"没有 user 消息的前缀"上会抛 `TemplateError`，改为跳过该轮而不是中断 |
| `models/qwen_vl/.../model.py` | packing 产生的 1D/2D `position_ids` 与 Qwen3-VL 的 3D MRoPE 不兼容时丢弃，由 `input_ids` 重建（27B VL 结构需要） |
| `models/qwen_vl/.../rope.py` | MRoPE 位置构建兼容零长度占位段和尾部填充 |
| `examples/conversion/adapter/export_adapter.py` | 导出 LoRA 时跳过训练中未挂 LoRA 的 MTP 层（否则加载 checkpoint 报 `KeyError`）；关闭 APEX 梯度融合和 fused RoPE |
| `scripts/conversion/gpu_backend.py` | HF↔Megatron 转换时关闭 APEX 梯度融合 |

**Megatron-LM / Megatron-Core**（5 个文件，`megatron-lm.patch`）：

| 文件 | 修改 |
|------|------|
| `transformer/transformer_config.py` | 新增配置项 `disable_thd_attention` |
| `transformer/attention.py` | `disable_thd_attention=True` 时 attention 忽略 packing 元数据，走 dense 计算；GDN 层仍按 packing 分段 |
| `ssm/gated_delta_net/gdn.py` | GDN 的 conv1d 和 gated delta rule 不再用 varlen kernel（Hopper 上反传 NaN），改为按 packing 段逐段 dense 计算后拼接 |
| `ssm/gated_delta_net/common.py` | `cu_seqlens` 优先使用逻辑长度（不含填充），去掉尾部零长度段，并校验不存在非正长度段 |
| `datasets/Makefile` | 编译 dataset C++ helper 时可指定 venv 的 Python（`PYTHON=...`） |

### 运行时补丁（`qwen35_lora/env/patches/`）

这部分不改上游源码，而是在 `train.py` 启动时以 monkey-patch 方式注入：

| 文件 | 作用 |
|------|------|
| `patch_sft_left_truncation.py` | 超长对话从左侧截断，保证 `last_turn` 模式下最后一轮 assistant 的监督目标不被截掉 |
| `patch_packing_left_truncate.py` | offline packing 时超长样本左截断而不是直接丢弃 |
| `patch_peft_ep_overlap.py` | LoRA + EP A2A overlap 时，冻结的 embedding 输出也能正常反传 |
| `patch_hybridep_pad_local.py` | HybridEP 用静态 `seq_length` 对齐，避免与 EP overlap 死锁 |
| `patch_te_sm103.py` | Transformer Engine attention 适配 GB300（sm103）并锁定 FlashAttention 版本 |
| `patch_te_nvte_pool.py` | 可选：把 TE 动态库重定向到更大 NVTETensor pool 的构建 |
| `bypass_mamba_tilelang.py` | 可选：绕过 `mamba_ssm` 对旧版 tilelang 的强依赖 |

---

## 目录结构

```text
megatron_qwen/
├── scripts/                     # 环境与数据工具
│   ├── setup_uv_env.sh          # 用 uv 创建 .venv 并安装依赖
│   ├── apply_megatron_patches.sh# 可选：从上游重新拉取 Megatron-Bridge 并打补丁
│   ├── download_hf_model.sh     # 通过 hf-mirror 下载 HF 模型
│   ├── hfd.sh                   # HF 多线程下载工具
│   └── convert_*_to_sft_jsonl.py# 各类评测轨迹 → SFT chat JSONL
├── Megatron-Bridge/             # 已打补丁的 Megatron-Bridge 源码（含 3rdparty/Megatron-LM）
├── patches/                     # 上述修改的 diff 形式（仅供查看 / 从上游重建）
│   ├── MEGATRON_BRIDGE_COMMIT   # 上游 commit
│   ├── megatron-bridge.patch
│   ├── MEGATRON_LM_COMMIT
│   └── megatron-lm.patch
├── qwen35_lora/                 # 训练主体
│   ├── run.sh                   # 一键启动（torchrun）
│   ├── train.py                 # 训练入口，调用 Bridge finetune()
│   ├── config.py                # 超参与路径（日常只改这个文件顶部）
│   ├── env/                     # 运行时补丁（TE / FA / HybridEP / PEFT 等）
│   ├── scripts/                 # 数据切分、HF→Megatron 转换、LoRA 合并、吞吐测试
│   └── docs/                    # SGLang 部署、LoRA 合并等说明
├── data/*.meta.json             # 使用过的数据集元信息（jsonl 本体不入库）
└── *.md                         # 完整指南与实验记录
```

运行过程中会在本地生成以下目录（均已写入 `.gitignore`）：`.venv/`、`models/`、`Megatron-Bridge/.../helpers_cpp*.so`（编译产物）、`data/*.jsonl`、`qwen35_lora/{data,checkpoints,logs,export,.cache}/`。

---

## 环境要求

- Linux，NVIDIA H200（sm90）；默认配置需要 **4 张空闲 GPU**
- CUDA 12.8（默认 `CUDA_HOME=/usr/local/cuda`）
- Python 3.12，已安装 [uv](https://github.com/astral-sh/uv) 和 `git`
- 磁盘：27B 模型的 HF 权重 + Megatron 权重约 110G，LoRA checkpoint 另需数十 G

---

## 快速开始

以下命令都在仓库根目录执行（下文用 `$PROJECT` 表示仓库根目录）。

### 第 1 步：确认 Megatron-Bridge 源码

仓库已自带打好补丁的 `Megatron-Bridge/`，克隆本仓库后无需额外操作。`run.sh`、`convert_hf_to_megatron.sh`、`setup_uv_env.sh` 默认都使用这个目录。

> 不要删除 `Megatron-Bridge/`。`setup_uv_env.sh` 如果找不到它，会自己克隆一份未打补丁的上游代码，训练会出问题。

可选：如果想从上游重新构建（例如核对改动），把 `Megatron-Bridge/` 移走后运行 `bash scripts/apply_megatron_patches.sh`，它会克隆上游指定 commit 到 `downloads/`、打上 `patches/` 中的补丁，并创建 `Megatron-Bridge` 软链接。

### 第 2 步：安装 Python 环境

```bash
bash scripts/setup_uv_env.sh
source .venv/bin/activate
```

安装内容：torch（cu128，阿里云镜像）、megatron-bridge / megatron-core、transformers、Transformer Engine 2.8.0，并编译 Megatron 数据集的 C++ helper。PyPI 默认走清华镜像，可以用 `PYPI_MIRROR` 覆盖。

可选：flash-attn、causal-conv1d 的预编译 wheel（单个文件超过 100MB，不入库）可放到 `wheels/` 下手动安装：

```bash
uv pip install wheels/flash_attn-2.8.1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
uv pip install wheels/causal_conv1d-1.5.2+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

安装完成后，脚本最后会打印 `torch ... avail True` 和 `megatron.bridge OK`。

### 第 3 步：准备 HF 基座模型

当前 `config.py` 的默认配置是 **Qwen3.8-27B（dense，VL 结构）LoRA**。先准备 HF 权重，然后修改 `qwen35_lora/config.py` 里的路径：

```python
HF_MODEL = "/path/to/qwen3.8_27b"   # 改成你本地的 HF 模型目录
```

如需下载模型（默认下载 Qwen3.5-35B-A3B-Base，可用环境变量改）：

```bash
HF_MODEL=Qwen/Qwen3.5-35B-A3B-Base \
LOCAL_DIR=$PROJECT/models/hf/Qwen3.5-35B-A3B-Base \
bash scripts/download_hf_model.sh
```

### 第 4 步：HF → Megatron 权重转换（一次性）

训练读取的是 Megatron `torch_dist` 格式，不是 HF safetensors。以 27B 为例（dense 模型用 `EP=1`）：

```bash
HF_MODEL=/path/to/qwen3.8_27b \
MEGATRON_PATH=$PROJECT/models/Qwen3.8-27B_megatron \
GPUS_PER_NODE=4 TP=1 PP=1 EP=1 \
bash qwen35_lora/scripts/convert_hf_to_megatron.sh
```

产物目录中应有 `latest_checkpointed_iteration.txt` 和 `iter_0000000/tokenizer/`。`config.py` 中的 `CKPT` 默认指向 `models/Qwen3.8-27B_megatron`，`TOKENIZER` 默认指向其中的 `iter_0000000/tokenizer`。

### 第 5 步：准备训练数据

原始数据格式：每行一个 JSON，至少包含 `messages`（chat 列表），可选 `tools`。

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "tools": null}
```

如果数据来自评测轨迹，可先用 `scripts/convert_*_to_sft_jsonl.py` 转换（各脚本 `--help` 查看参数；默认输入路径是集群路径，需要用参数覆盖）。

把原始 JSONL 放到 `data/` 下，再切分成训练集和验证集。默认数据集是 `fea_bench_glm53_lite200`：

```bash
cd qwen35_lora
python3 scripts/prepare_data.py \
  --src ../data/fea_bench_glm53_lite200.jsonl \
  --out-dir ../data/fea_bench_glm53_lite200 \
  --val-ratio 0.01 --seed 42
```

输出 `training.jsonl`、`validation.jsonl`、`prepare_meta.json`。使用其他数据集时，修改 `config.py` 的 `RAW_DATA` / `DATA_DIR`，并同步修改 `run.sh` 里的 `DATA_DIR`（或训练时传 `--dataset-root`）。

### 第 6 步：冒烟测试

```bash
cd qwen35_lora
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash run.sh smoke
```

smoke 会自动从 `RAW_DATA` 抽 128 条样本到 `qwen35_lora/data/smoke`，seq=8192，跑 10 个 iter。日志看到 loss 正常输出、程序正常退出即说明环境可用。

### 第 7 步：正式训练

```bash
cd qwen35_lora
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash run.sh
```

默认配置：27B LoRA，`TP=2 PP=1 EP=1`，seq=8192，GBS=16，LoRA dim/alpha = 32/64，LoRA+ ratio 16，LR=2e-5，`loss_mode=last_turn`，3 epoch，每 50 step 做一次 eval 并保存。

- 日志：`qwen35_lora/logs/<mode>_<时间戳>.log`
- checkpoint：`config.py` 中的 `SAVE_DIR`（默认 `qwen35_lora/checkpoints/lora_27b_feabench_lite200_lastturn_gbs16/`）
- TensorBoard：`<SAVE_DIR>/tb_logs`

> `run.sh` 未设置 `CUDA_VISIBLE_DEVICES` 时默认使用 `4,5,6,7`，请按机器实际空闲卡设置。

---

## 运行模式与常用参数

`run.sh` 的第一个参数是模式，`--` 之后的参数会传给 `train.py`，覆盖 `config.py` 的默认值。

| 命令 | 作用 |
|------|------|
| `bash run.sh` | 正式训练（LoRA） |
| `bash run.sh smoke` | 短训练冒烟，检查环境 |
| `bash run.sh bench` | 固定 8 iter 测吞吐，不保存 checkpoint |
| `bash run.sh profile` | torch.profiler 采样（第 3～5 步） |
| `bash run.sh -- --full-sft` | 全量微调（LR=5e-6，seq=4096，selective 重算） |
| `bash run.sh smoke -- --full-sft` | 全量微调冒烟（seq=2048） |

常用覆盖参数：

```bash
bash run.sh bench -- --iters 8 --seq 4096 --gbs 16 --precision bf16 --attn-backend fused
bash run.sh -- --tp 2 --pp 1 --ep 1 --seq 8192 --gbs 16
bash run.sh -- --dataset-root /path/to/data_dir --save-dir /path/to/ckpt_dir
bash run.sh -- --load-checkpoint /path/to/ckpt_dir   # 断点续训
```

完整参数见 `python qwen35_lora/train.py --help`。约束：GPU 数量 `NPROC` 必须 ≥ `TP×PP×EP` 且能被它整除；GBS 必须能被数据并行度整除。

如需多配置对比吞吐：`bash qwen35_lora/scripts/bench_mfu_sweep.sh`。

---

## 训练后：导出与部署

### 导出 LoRA 为 HF PEFT 格式

```bash
cd Megatron-Bridge
export PYTHONPATH="src:3rdparty/Megatron-LM:${PYTHONPATH:-}"

python -m torch.distributed.run --nproc_per_node=4 \
  examples/conversion/adapter/export_adapter.py \
  --hf-model-path /path/to/qwen3.8_27b \
  --lora-checkpoint $PROJECT/qwen35_lora/checkpoints/<run_name>/iter_XXXXXXX \
  --output $PROJECT/qwen35_lora/export/lora_hf \
  --trust-remote-code --dtype bf16 \
  --exclude-adapter-base-prefix mtp.layers \
  --tp 2 --pp 1 --ep 1
```

`--tp/--pp/--ep` 与训练保存时的并行配置保持一致。

### 合并 LoRA 到基座

参考 `qwen35_lora/scripts/merge_lora_lcb1k.py` / `merge_lora_swe1k.py`。脚本顶部的 `BASE`、`ADAPTER`、`OUT` 是写死的集群路径，使用前需要改成自己的路径。

### 全量微调导出

全量微调没有 adapter，使用 `Megatron-Bridge/scripts/conversion/convert.sh export` 导出完整 HF 权重，详见 `complete_guide_full_sft.md`。

### SGLang 部署

见 `qwen35_lora/docs/launch_lcb1k_sglang.md` 与 `qwen35_lora/docs/sglang_lora_merge.md`。

---

## 切换模型

- **Qwen3.5 / Qwen3.6-35B-A3B（MoE）**：修改 `config.py` 中的 `HF_MODEL`、`CKPT`，并把并行改为 `TP, PP, EP = 1, 1, 4`（4 卡）；MoE 可开启 `HYBRIDEP` / `EP_OVERLAP`。转换时 `convert_hf_to_megatron.sh` 的默认参数就是 35B-A3B。
- **Qwen3.8-27B（dense）**：当前默认配置，必须 `EP=1`，不支持 HybridEP / EP overlap。

---

## 相关文档

| 文档 | 内容 |
|------|------|
| `complete_guide_from_zero.md` | 从零开始的完整 LoRA 流程说明（环境、转换、数据、packing、训练、导出） |
| `complete_guide_full_sft.md` | 全量微调说明（与 LoRA 的差别、并行与显存、导出） |
| `8.21 - 8.25 megatron-SFT-1.md` | 实验记录 |
| `qwen35_lora/docs/` | SGLang 部署、LoRA 合并、ConditionalGeneration 合并说明 |

---

## 常见问题

| 现象 | 处理 |
|------|------|
| `WORLD_SIZE=... 必须能被 TP*PP*EP 整除` | 调整 `NPROC` 或 `--tp/--pp/--ep` |
| `missing .../training.jsonl` | 先执行第 5 步的 `prepare_data.py` |
| 预训练权重加载失败 | `CKPT` 必须是 `torch_dist` 目录；27B 是 VL 结构，需用本仓库转换脚本产出的 `language_model.*` 权重 |
| 27B 在 16k / 32k 序列 OOM | 4×H200 下保持 seq=8192 + `RECOMPUTE="full"` |
| packing 下 loss 出现 NaN | 保持 `PACKING_ATTN_THD = False` |
| 从上游重建时补丁打不上 | 确认 `downloads/Megatron-Bridge-src` 是干净的上游仓库，删除后重新运行 `apply_megatron_patches.sh` |
