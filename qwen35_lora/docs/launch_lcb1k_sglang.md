# Qwen3.8-27B-lcb1k SGLang 启动记录

## 当前模型

- 合并模型：`/kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/export/merged_hf_lcb1k`
- Docker 容器：`sglang-qwen38-27b-lcb1k`
- 镜像：`lmsysorg/sglang:qwen38flashnext`
- GPU：物理卡 4–7，容器内映射为 0–3
- TP：4
- 服务地址：`http://0.0.0.0:30001/v1`
- served model：`Qwen3.8-27B-lcb1k`

## 当前采用的启动参数

```bash
python3 -m sglang.launch_server \
  --model-path /model \
  --served-model-name Qwen3.8-27B-lcb1k \
  --tp 4 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --context-length 262144 \
  --chunked-prefill-size 4096 \
  --attention-backend fa3 \
  --linear-attn-prefill-backend flashinfer \
  --linear-attn-decode-backend flashinfer \
  --linear-attn-verify-backend triton \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30001
```

CUDA graph 保持开启：没有使用 `--disable-cuda-graph`。

## 容器启动方式

```bash
docker run -d \
  --name sglang-qwen38-27b-lcb1k \
  --gpus '"device=4,5,6,7"' \
  --network host \
  --shm-size 32g \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -v /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/export/merged_hf_lcb1k:/model:ro \
  -v /kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/logs/sglang_lcb1k:/logs \
  lmsysorg/sglang:qwen38flashnext \
  bash -lc 'sleep infinity'
```

进入已存在的容器后，执行上面的 `launch_server` 命令，并将日志重定向到：

`/logs/sglang-qwen38-27b-lcb1k.log`

## 停止

```bash
docker stop -t 30 sglang-qwen38-27b-lcb1k
```

本次已停止评测进程和该模型容器；GPU 4–7 已释放。评测产物保留在：

`/kwkj-k8s/llm_team/ycm/benchmark/swebench/swe_runs/ycm_qwen38_lcb1k_verified_full_ctx256k`

## GPU 0–3 当前启动变体

前四张卡原先的 `qwen38-flash-next-ties-sglang-4card` 已停止。
由于宿主机 `30001` 被 `qwen38-tb226-ckpt2750` 占用，本模型当前使用端口
`30002`，容器名为 `sglang-qwen38-27b-lcb1k-gpu0-3`。

当前变体使用：

- `--mem-fraction-static 0.85`
- `--context-length 262144`
- CUDA graph 开启（不使用 `--disable-cuda-graph`）
- GPU 0–3，TP=4，FP8

当前接口：

`http://127.0.0.1:30002/v1`
