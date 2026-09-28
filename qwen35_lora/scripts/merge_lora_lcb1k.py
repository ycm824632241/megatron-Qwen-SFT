#!/usr/bin/env python3
"""Merge LiveCodeBench 1K LoRA into Qwen3.8-27B ConditionalGeneration HF weights."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer

BASE = Path("/kwkj-k8s/llm_team/model/qwen3.8_27b")
ADAPTER = Path("/kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/export/lora_hf_lcb1k")
OUT = Path("/kwkj-k8s/llm_team/ycm/megatron_qwen35/qwen35_lora/export/merged_hf_lcb1k")

EXTRA = [
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "configuration.json",
]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    print("base:", BASE, flush=True)
    print("adapter:", ADAPTER, flush=True)
    print("out:", OUT, flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    max_memory = {i: "90GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForImageTextToText.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("loaded:", type(model).__name__, flush=True)
    if "ConditionalGeneration" not in type(model).__name__:
        raise SystemExit(f"refusing to merge: got {type(model).__name__}")

    model = PeftModel.from_pretrained(model, str(ADAPTER))
    print("peft loaded", flush=True)
    model = model.merge_and_unload()
    print("merged", flush=True)

    model.save_pretrained(OUT, max_shard_size="5GB")
    tok.save_pretrained(OUT)
    for name in EXTRA:
        src = BASE / name
        if src.exists():
            shutil.copy2(src, OUT / name)
            print("copied", name, flush=True)

    arch = json.loads((OUT / "config.json").read_text())["architectures"]
    print("architectures:", arch, flush=True)
    if arch != ["Qwen3_5ForConditionalGeneration"]:
        raise SystemExit(f"bad architectures: {arch}")
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
