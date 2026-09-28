#!/usr/bin/env python3
"""数据预处理：原始 chat JSONL → Bridge ``GPTSFTDatasetConfig`` 认的切分文件。

输入
----
每行一个 JSON，至少含 ``messages``（list）。可选 ``tools``（字符串或 list）。

输出（写到 ``--out-dir``）
------------------------
- ``training.jsonl`` / ``validation.jsonl``
- ``prepare_meta.json``（扫描统计）

每行形态（Bridge chat schema）::

    {
      "messages": [{"role": "...", "content": "..."}, ...],
      "tools": [{"type": "function", "function": {...}}, ...]  # 或 null
    }

说明
----
- 本脚本只做：过滤空 messages、规范化 tools、可选长度过滤、train/val 划分。
- **不算** loss mask、**不做** packing（那是训练时 Bridge 的事）。
- ``run.sh`` 在发现已有 ``training.jsonl`` 时会跳过本脚本。

用法::

    python scripts/prepare_data.py --out-dir data/full --val-ratio 0.01
    python scripts/prepare_data.py --out-dir data/smoke --max-seq-length 8192 --max-train 128
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import RAW_DATA, TOKENIZER  # noqa: E402


def normalize_tools(tools):
    """把业务侧 tools 收敛成 OpenAI / HF chat-template 常见的 list[dict]。

    接受：
    - ``None``
    - JSON 字符串（会 loads）
    - 已经是 ``[{"type":"function","function":{...}}, ...]``
    - 裸 function dict 列表（自动包一层 type/function）
    """
    if tools is None:
        return None
    if isinstance(tools, (str, bytes)):
        tools = json.loads(tools)
    if not isinstance(tools, list):
        raise TypeError(f"tools must be list, got {type(tools)}")
    out = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise TypeError(f"tool entry must be dict, got {type(tool)}")
        if tool.get("type") == "function" and "function" in tool:
            out.append(tool)
        else:
            out.append({"type": "function", "function": tool})
    return out


def normalize_messages(messages: list) -> list | None:
    """Qwen chat template 要求前缀里必须有 user。丢掉首个 user 之前的 assistant/tool。"""
    first_user = next(
        (i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"),
        None,
    )
    if first_user is None:
        return None
    head = [m for m in messages[:first_user] if isinstance(m, dict) and m.get("role") == "system"]
    tail = [m for m in messages[first_user:] if isinstance(m, dict)]
    out = head + tail
    if not any(m.get("role") == "assistant" for m in out):
        return None
    return out


def _message_chars(messages: list) -> int:
    """粗估 messages 文本长度，超长样本可在 tokenize 前丢掉。"""
    n = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content", "")
        if isinstance(content, str):
            n += len(content)
        elif isinstance(content, list):
            n += sum(len(str(x)) for x in content)
        else:
            n += len(str(content))
    return n


def iter_rows(
    src: Path,
    *,
    tokenizer=None,
    max_seq_length: int = 0,
    scan_limit: int = 0,
    max_keep: int = 0,
) -> tuple[list[dict], dict]:
    """扫描原始 JSONL，返回可用行 + 统计。"""
    rows: list[dict] = []
    scanned = skipped_len = skipped_bad = 0
    # 即便 1 token ≈ 8 字符，超过此值也必然 > max_seq_length，不必整段 encode。
    char_limit = max_seq_length * 8 if max_seq_length > 0 else 0

    with src.open() as f:
        for line in f:
            if scan_limit and scanned >= scan_limit:
                break
            if max_keep and len(rows) >= max_keep:
                break
            scanned += 1
            try:
                raw = json.loads(line)
                messages = raw.get("messages")
                if not isinstance(messages, list) or not messages:
                    skipped_bad += 1
                    continue
                messages = normalize_messages(messages)
                if messages is None:
                    skipped_bad += 1
                    continue
                try:
                    tools = normalize_tools(raw.get("tools"))
                except Exception:
                    skipped_bad += 1
                    continue

                # smoke 等场景：用 chat template 估长度，丢掉过短/过长样本。
                if tokenizer is not None and max_seq_length > 0:
                    if char_limit and _message_chars(messages) > char_limit:
                        skipped_len += 1
                        continue
                    try:
                        text = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                            tools=tools,
                        )
                        # 截到 max+1：超长样本不必编到 26 万 token，也避免 transformers 告警。
                        n = len(
                            tokenizer.encode(
                                text,
                                add_special_tokens=False,
                                truncation=True,
                                max_length=max_seq_length + 1,
                            )
                        )
                    except Exception:
                        skipped_len += 1
                        continue
                    if n <= 8 or n > max_seq_length:
                        skipped_len += 1
                        continue

                rows.append({"messages": messages, "tools": tools})
            finally:
                if scanned % 50 == 0:
                    print(
                        f"[prepare] scanned={scanned} kept={len(rows)} "
                        f"skipped_len={skipped_len} skipped_bad={skipped_bad}",
                        flush=True,
                    )

    stats = {
        "scanned": scanned,
        "kept": len(rows),
        "skipped_len": skipped_len,
        "skipped_bad": skipped_bad,
    }
    return rows, stats


def split_train_val(
    rows: list[dict],
    *,
    val_ratio: float,
    seed: int,
    max_train: int = 0,
    max_val: int = 0,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(1, int(round(len(rows) * val_ratio))) if val_ratio > 0 else 0
    n_val = min(n_val, max(0, len(rows) - 1))
    val, train = rows[:n_val], rows[n_val:]
    if max_train > 0:
        train = train[:max_train]
    if max_val > 0:
        val = val[:max_val]
    return train, val


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Bridge memmap 会复用同名 .idx.npy/.idx.info；旧索引切到新 jsonl 会 JSONDecodeError。
    for sidecar in (Path(f"{path}.idx.npy"), Path(f"{path}.idx.info")):
        sidecar.unlink(missing_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=Path(RAW_DATA), help="原始 chat JSONL")
    p.add_argument("--out-dir", type=Path, required=True, help="写出 training/validation.jsonl 的目录")
    p.add_argument("--tokenizer", type=Path, default=Path(TOKENIZER), help="长度过滤用 tokenizer")
    p.add_argument("--val-ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-seq-length", type=int, default=0, help=">0 时丢弃超长样本（smoke）")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--scan-limit", type=int, default=0, help=">0 时最多扫描前 N 行")
    args = p.parse_args()

    tokenizer = None
    if args.max_seq_length > 0:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)

    max_keep = 0
    if args.max_train or args.max_val:
        # 多留一倍给 val 划分；够数就停，不必扫完 scan-limit。
        max_keep = max(args.max_train + args.max_val, 1) * 2

    rows, stats = iter_rows(
        args.src,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        scan_limit=args.scan_limit,
        max_keep=max_keep,
    )
    if not rows:
        raise SystemExit(f"No usable rows from {args.src}")

    train, val = split_train_val(
        rows,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_train=args.max_train,
        max_val=args.max_val,
    )

    write_jsonl(args.out_dir / "training.jsonl", train)
    write_jsonl(args.out_dir / "validation.jsonl", val)

    meta = {
        "src": str(args.src),
        **stats,
        "train": len(train),
        "val": len(val),
        "val_ratio": args.val_ratio,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
    }
    (args.out_dir / "prepare_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
