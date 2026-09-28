#!/usr/bin/env python3
"""Convert OpenAI MRCR eval traces to Megatron chat SFT JSONL.

Rules (align with ``ycm/doc/Megatron数据格式速记.md`` and
``complete_guide_from_zero.md`` §5.2):

- Keep only correct traces: ``score >= --min-score`` (default ``1.0``).
- Keep only ``messages`` (strip eval metadata).
- Last assistant must equal gold ``answer`` (score=1 already implies this).
- Merge consecutive same-role turns (MRCR starts with two ``user`` turns).
- Historical assistant: ``loss=false``; last assistant: ``loss=true``.
- Do not expand intermediate needle turns: those are context, not targets.

Writes a single merged file ``data/mrcr.jsonl``. Duplicate conversations
(e.g. smoke ⊂ n2) are dropped.

Usage::

    python3 scripts/convert_mrcr_results_to_sft_jsonl.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

SRC_DIR = Path("/kwkj-k8s/llm_team/ycm/Datasets/MRCR/results")
OUT_DIR = Path("/kwkj-k8s/llm_team/ycm/megatron_qwen35/data")
MERGED_NAME = "mrcr.jsonl"

DEFAULT_SOURCES = [
    "DeepSeek-V4-Flash-0731_n2.jsonl",
    "DeepSeek-V4-Flash-0731_n4.jsonl",
    "DeepSeek-V4-Flash-0731_n8.jsonl",
    "DeepSeek-V4-Flash-0731_n2_1m_more_20260828_144830.jsonl",
    "DeepSeek-V4-Flash-0731_smoke.jsonl",
]


def merge_consecutive(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role not in ALLOWED_ROLES:
            continue
        content = msg.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)
        if merged and merged[-1]["role"] == role:
            prev = merged[-1]["content"] or ""
            if prev and content:
                merged[-1]["content"] = prev.rstrip() + "\n\n" + content.lstrip()
            else:
                merged[-1]["content"] = prev or content
        else:
            merged.append({"role": role, "content": content})
    return merged


def apply_loss(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_asst = max(
        (i for i, m in enumerate(messages) if m["role"] == "assistant"),
        default=None,
    )
    out: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        item = {"role": msg["role"], "content": msg.get("content") or ""}
        if item["role"] == "assistant":
            item["loss"] = i == last_asst
        out.append(item)
    return out


def is_correct(rec: dict[str, Any], min_score: float) -> bool:
    try:
        score = float(rec.get("score") or 0.0)
    except (TypeError, ValueError):
        return False
    if score < min_score:
        return False
    gold = rec.get("answer")
    if not isinstance(gold, str) or not gold:
        return False
    msgs = rec.get("messages") or []
    if not msgs or not isinstance(msgs[-1], dict):
        return False
    last = msgs[-1].get("content") or ""
    if min_score >= 1.0 and last != gold:
        return False
    return True


def convert_record(rec: dict[str, Any], *, min_score: float) -> dict[str, Any] | None:
    if not is_correct(rec, min_score):
        return None

    raw = rec.get("messages") or []
    if not isinstance(raw, list) or not raw:
        return None

    gold = rec.get("answer")
    if not isinstance(gold, str) or not gold:
        return None

    msgs: list[dict[str, Any]] = []
    for m in raw:
        if isinstance(m, dict):
            msgs.append(m)

    if msgs and msgs[-1].get("role") == "assistant":
        msgs[-1] = {**msgs[-1], "content": gold, "role": "assistant"}
    else:
        msgs.append({"role": "assistant", "content": gold})

    msgs = merge_consecutive(msgs)
    if not any(m["role"] == "user" for m in msgs):
        return None
    if not any(m["role"] == "assistant" for m in msgs):
        return None
    if not (msgs[-1]["role"] == "assistant" and (msgs[-1].get("content") or "").strip()):
        return None

    return {"messages": apply_loss(msgs)}


def sample_key(row: dict[str, Any]) -> str:
    payload = json.dumps(row["messages"], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def convert_sources(
    src_dir: Path,
    dst: Path,
    sources: list[str],
    *,
    min_score: float,
) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    per_src: list[dict[str, Any]] = []
    kept = skipped = dup = 0
    n_msgs: Counter[int] = Counter()

    with dst.open("w", encoding="utf-8") as fout:
        for src_name in sources:
            src = src_dir / src_name
            if not src.exists():
                print(f"[skip] missing {src}")
                continue
            src_kept = src_skip = src_dup = 0
            with src.open(encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        src_skip += 1
                        continue
                    out = convert_record(rec, min_score=min_score)
                    if out is None:
                        skipped += 1
                        src_skip += 1
                        continue
                    key = sample_key(out)
                    if key in seen:
                        dup += 1
                        src_dup += 1
                        continue
                    seen.add(key)
                    fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                    kept += 1
                    src_kept += 1
                    n_msgs[len(out["messages"])] += 1
            per_src.append(
                {
                    "src": str(src),
                    "kept": src_kept,
                    "skipped": src_skip,
                    "dup": src_dup,
                }
            )
            print(
                f"[src] {src_name}: kept={src_kept} skipped={src_skip} dup={src_dup}"
            )

    return {
        "dst": str(dst),
        "min_score": min_score,
        "kept": kept,
        "skipped": skipped,
        "dup": dup,
        "per_src": per_src,
        "n_messages": dict(sorted(n_msgs.items())),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=Path, default=SRC_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--out-name", type=str, default=MERGED_NAME)
    p.add_argument(
        "--min-score",
        type=float,
        default=1.0,
        help="drop traces below this MRCR score (1.0 = exact match)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dst = args.out_dir / args.out_name
    info = convert_sources(
        args.src_dir, dst, DEFAULT_SOURCES, min_score=args.min_score
    )
    print(
        f"[ok] {dst.name}: kept={info['kept']} skipped={info['skipped']} "
        f"dup={info['dup']} min_score={args.min_score}"
    )
    meta = args.out_dir / "mrcr_sft.meta.json"
    meta.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[meta] {meta}")


if __name__ == "__main__":
    main()
