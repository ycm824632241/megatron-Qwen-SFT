#!/usr/bin/env python3
"""把 SWE-bench-Live mini-swe-agent ``*.traj.json`` 转成 Megatron chat SFT JSONL。

对齐 ``complete_guide_from_zero.md`` §5.2 与 ``ycm/doc/Megatron数据格式速记.md``：
- 默认只保留 ``exit_status=Submitted`` 且 submission 非空的轨迹
- 丢掉 ``exit`` 等非标准 role
- 把 ``reasoning_content`` / ``tool_calls`` 渲染进 ``assistant.content``
- 按每个 assistant 轮切成一条样本：历史 assistant ``loss=false``，目标轮 ``loss=true``
- ``tools`` 为 bash 扁 schema 的 JSON 字符串

用法::

    python3 /kwkj-k8s/llm_team/ycm/megatron_qwen35/scripts/convert_swe_live_traj_to_sft_jsonl.py \\
      --gen-dir /kwkj-k8s/llm_team/ycm/Datasets-coding/SWE-bench-Live/logs/glm53_lite_v2/gen \\
      --out /kwkj-k8s/llm_team/ycm/megatron_qwen35/data/swe_live_glm53.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 复用 tau3 转换里的渲染 / 按轮展开
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_tau3_results_to_sft_jsonl import (  # noqa: E402
    convert_messages,
    expand_by_assistant_turns,
)

BASH_TOOLS = json.dumps(
    [
        {
            "name": "bash",
            "description": "Run a bash command in the SWE-bench sandbox (/testbed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    }
                },
                "required": ["command"],
            },
        }
    ],
    ensure_ascii=False,
)

DEFAULT_GEN = Path(
    "/kwkj-k8s/llm_team/ycm/Datasets-coding/SWE-bench-Live/logs/glm53_lite_v2/gen"
)
DEFAULT_OUT = Path("/kwkj-k8s/llm_team/ycm/megatron_qwen35/data/swe_live_glm53.jsonl")


def load_preds(gen_dir: Path) -> dict[str, Any]:
    p = gen_dir / "preds.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def iter_traj(gen_dir: Path) -> list[Path]:
    return sorted(gen_dir.glob("*/*.traj.json"))


def keep_traj(
    traj: dict[str, Any],
    preds: dict[str, Any],
    *,
    include_failed: bool,
) -> bool:
    info = traj.get("info") or {}
    status = str(info.get("exit_status") or "")
    iid = traj.get("instance_id") or ""
    submission = (info.get("submission") or "").strip()
    pred_patch = ""
    if iid in preds:
        pred_patch = ((preds[iid] or {}).get("model_patch") or "").strip()
    has_patch = bool(submission or pred_patch)
    if include_failed:
        return bool(traj.get("messages"))
    return status == "Submitted" and has_patch


def convert_one(
    traj: dict[str, Any],
    *,
    keep_turns: str,
    keep_meta: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw = [m for m in (traj.get("messages") or []) if isinstance(m, dict)]
    messages = convert_messages(raw, system_prompt=None)
    if not any(m["role"] == "assistant" for m in messages):
        return [], {"skipped_no_assistant": 1}

    info = traj.get("info") or {}
    meta = {
        "instance_id": traj.get("instance_id"),
        "exit_status": info.get("exit_status"),
        "mini_version": info.get("mini_version"),
        "source": "swe-bench-live",
    }
    samples, skipped = expand_by_assistant_turns(
        messages,
        tools_str=BASH_TOOLS,
        meta=meta,
        keep_turns=keep_turns,
    )
    if not keep_meta:
        for s in samples:
            s.pop("meta", None)
    return samples, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="也转换非 Submitted / 空 patch 轨迹",
    )
    parser.add_argument("--keep-meta", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--keep-turns",
        choices=("all", "tool_call", "skip_greeting"),
        default="skip_greeting",
    )
    args = parser.parse_args(argv)

    preds = load_preds(args.gen_dir)
    paths = iter_traj(args.gen_dir)
    stats: dict[str, Any] = {
        "n_traj_files": len(paths),
        "kept_traj": 0,
        "skipped_filter": 0,
        "skipped_no_assistant": 0,
        "skipped_greeting": 0,
        "skipped_no_tool_call": 0,
        "n_samples": 0,
        "keep_turns": args.keep_turns,
        "include_failed": args.include_failed,
    }
    rows: list[dict[str, Any]] = []

    for path in paths:
        traj = json.loads(path.read_text())
        if not keep_traj(traj, preds, include_failed=args.include_failed):
            stats["skipped_filter"] += 1
            continue
        samples, skipped = convert_one(
            traj, keep_turns=args.keep_turns, keep_meta=args.keep_meta
        )
        stats["skipped_greeting"] += skipped.get("skipped_greeting", 0)
        stats["skipped_no_tool_call"] += skipped.get("skipped_no_tool_call", 0)
        if not samples:
            stats["skipped_no_assistant"] += 1
            continue
        rows.extend(samples)
        stats["kept_traj"] += 1
        stats["n_samples"] += len(samples)

    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    stats["n_samples_written"] = len(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps({"out": str(args.out), "gen_dir": str(args.gen_dir), **stats}, indent=2, ensure_ascii=False)
        + "\n"
    )
    print(json.dumps({"out": str(args.out), **stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
