# Merge 后启动错误：按 ConditionalGeneration 重新合并

## 背景

动态 LoRA（`--enable-lora`）在 SGLang `v0.5.16` 上不可用，因此需要把 adapter merge 进基座后再起服。

若 merge 时误用 `AutoModelForCausalLM`，写出的 `merged_hf` 架构会变成 `Qwen3_5MoeForCausalLM`，SGLang 无法加载。

| 路径 | 说明 |
|------|------|
| adapter | `qwen35_lora/export/lora_hf/` |
| Instruct 基座 | `/kwkj-k8s/llm_team/models/Qwen3.5-35B-A3B` |
| 合并产物 | `qwen35_lora/export/merged_hf/` |

---

## 现象

```text
ValueError: Qwen3_5MoeForCausalLM has no SGlang implementation
and the Transformers implementation is not compatible with SGLang.
```

---

## 原因

| | 官方 Instruct | 错误的 merged |
|--|----------------|---------------|
| `architectures` | `Qwen3_5MoeForConditionalGeneration` | `Qwen3_5MoeForCausalLM` |
| `model_type` | `qwen3_5_moe` | `qwen3_5_moe_text` |
| SGLang | 认识 | **不认识** |

`AutoModelForCausalLM` 会把 Qwen3.5 收成纯文本 CausalLM，并重写 `config.json`。  
SGLang 对 Qwen3.5 只认 **ConditionalGeneration**。

只改 `config.json` 不够，需要按正确类重新 merge。

---

## 正确做法：重复合并

### 1. 删除旧目录

避免新旧 shard / `index.json` 混用：

```bash
rm -rf /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/export/merged_hf
```

### 2. 按 ConditionalGeneration 重新 merge

```bash
source /kwkj-k8s/llm_team/ycm/megatron_qwen35/.venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3 python - <<'PY'
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer
from pathlib import Path
import shutil
import json

base = "/kwkj-k8s/llm_team/models/Qwen3.5-35B-A3B"
adapter = "/kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/export/lora_hf"
out = "/kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/export/merged_hf"

tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
# 禁止 AutoModelForCausalLM，否则会变成 Qwen3_5MoeForCausalLM
model = AutoModelForImageTextToText.from_pretrained(
    base, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True
)
print("loaded:", type(model).__name__)  # 应为 *ConditionalGeneration*

model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, max_shard_size="5GB")
tok.save_pretrained(out)

for name in [
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
]:
    src = Path(base) / name
    if src.exists():
        shutil.copy2(src, Path(out) / name)

print("merged ->", out)
print("architectures:", json.loads((Path(out) / "config.json").read_text())["architectures"])
PY
```

若环境没有 `AutoModelForImageTextToText`，改用：

```python
from transformers import Qwen3_5MoeForConditionalGeneration
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
    base, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True
)
```

### 3. 验收

`merged_hf/config.json` 必须是：

```json
"architectures": ["Qwen3_5MoeForConditionalGeneration"]
```

说明：分片是 2 个还是 16 个只影响打包切法；总大小约 65–67G 即可。

---

## Merge 完成后起服（不要开 LoRA）

```bash
docker rm -f sglang-qwen35-35b-a3b 2>/dev/null

docker run -d \
  --name sglang-qwen35-35b-a3b \
  --gpus '"device=4,5,6,7"' \
  --network host \
  --shm-size 32g \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -v /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/export/merged_hf:/model:ro \
  -v /kwkj-k8s/llm_team/hy/logs/sglang-qwen35-35b-a3b:/logs \
  lmsysorg/sglang:v0.5.16 \
  bash -lc 'sleep infinity'

docker exec -d sglang-qwen35-35b-a3b bash -lc '
exec python3 -m sglang.launch_server \
  --model-path /model \
  --served-model-name Qwen3.5-35B-A3B-rho \
  --tp 1 \
  --dp 4 \
  --mem-fraction-static 0.90 \
  --mamba-radix-cache-strategy extra_buffer \
  --context-length 262144 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --host 0.0.0.0 \
  --port 30001 \
  > /logs/sglang-qwen35-35b-a3b-server.log 2>&1
'
```

```bash
tail -f /kwkj-k8s/llm_team/hy/logs/sglang-qwen35-35b-a3b/sglang-qwen35-35b-a3b-server.log
```

```bash
curl http://127.0.0.1:30001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.5-35B-A3B-rho",
    "messages": [{"role":"user","content":"你好"}],
    "max_tokens": 256
  }'
```

---

## 备注

- 重新 merge 前务必 `rm -rf merged_hf`。
- 禁止再用 `AutoModelForCausalLM` 做本次合并。
- LoRA 在 Base 上训练；合并到 Instruct 可起服，效果不完全等于「Base + LoRA」。
