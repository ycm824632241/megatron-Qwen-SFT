#!/usr/bin/env python3
"""把 FEA-Bench mini-swe-agent ``*.traj.json`` 转成 Megatron chat SFT JSONL。

对齐 ``complete_guide_from_zero.md`` §5.2 与 SWE-Live 转换脚本：
- **默认只保留 harness ``resolved_ids``**（测通用例通过），不是 ``Submitted``
- ``Submitted`` 只说明交了 patch；要用 ``--allow-unresolved`` 才会按 Submitted 放行
- 跳过 ``gen_failed_*`` 重试失败目录
- 丢掉 ``exit`` 等非标准 role
- 把 ``reasoning_content`` / ``tool_calls`` 渲染进 ``assistant.content``
- 按每个 assistant 轮切成一条样本：历史 assistant ``loss=false``，目标轮 ``loss=true``
- ``tools`` 为 bash 扁 schema 的 JSON 字符串

用法::

    python3 /kwkj-k8s/llm_team/ycm/megatron_qwen35/scripts/convert_fea_bench_traj_to_sft_jsonl.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_swe_live_traj_to_sft_jsonl import keep_traj, load_preds  # noqa: E402
from convert_tau3_results_to_sft_jsonl import (  # noqa: E402
    convert_messages,
    expand_by_assistant_turns,
)

BASH_TOOLS = json.dumps(
    [
        {
            "name": "bash",
            "description": "Run a bash command in the FEA-Bench sandbox (/testbed).",
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
    "/kwkj-k8s/llm_team/ycm/Datasets-coding/FEA-Bench/feabench-data/"
    "experiments/agent/glm53_lite200/gen"
)
DEFAULT_OUT = Path("/kwkj-k8s/llm_team/ycm/megatron_qwen35/data/fea_bench_glm53_lite200.jsonl")
DEFAULT_REPORT = Path(
    "/kwkj-k8s/llm_team/ycm/Datasets-coding/FEA-Bench/feabench-data/"
    "experiments/agent/glm53_lite200/grade/openai__GLM5.3.glm53_lite200.json"
)


def load_resolved_ids(report_path: Path) -> set[str]:
    """SWE-bench ``run_evaluation`` 汇总 JSON 的 ``resolved_ids``。"""
    data = json.loads(report_path.read_text())
    ids = data.get("resolved_ids")
    if not isinstance(ids, list):
        raise SystemExit(f"no resolved_ids in {report_path}")
    return {str(x) for x in ids}


def iter_traj(gen_dir: Path) -> list[Path]:
    """只取 ``gen/<instance>/<instance>.traj.json``，不含 gen_failed_*。"""
    out: list[Path] = []
    for path in sorted(gen_dir.glob("*/*.traj.json")):
        if "gen_failed" in path.parts:
            continue
        out.append(path)
    return out


def convert_one(
    traj: dict[str, Any],
    *,
    keep_turns: str,
    keep_meta: bool,
    instance_id_fallback: str = "",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw = [m for m in (traj.get("messages") or []) if isinstance(m, dict)]
    messages = convert_messages(raw, system_prompt=None)
    if not any(m["role"] == "assistant" for m in messages):
        return [], {"skipped_no_assistant": 1}

    info = traj.get("info") or {}
    iid = traj.get("instance_id") or instance_id_fallback
    meta = {
        "instance_id": iid,
        "exit_status": info.get("exit_status"),
        "mini_version": info.get("mini_version"),
        "source": "fea-bench",
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
        "--resolved-report",
        type=Path,
        default=DEFAULT_REPORT,
        help="SWE-bench 评测汇总 JSON（含 resolved_ids）",
    )
    parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="不按 resolved 过滤，退回 Submitted+patch（不推荐做 SFT）",
    )
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
    resolved: set[str] | None = None
    if not args.allow_unresolved:
        if not args.resolved_report.exists():
            raise SystemExit(
                "FEA-Bench SFT 默认只保留 resolved 轨迹，但评测报告不存在：\n"
                f"  {args.resolved_report}\n"
                "glm53_lite200 的 grade 未跑通（log 里 ModuleNotFoundError）。\n"
                "请先 run_evaluation 得到 resolved_ids，或显式 --allow-unresolved。"
            )
        resolved = load_resolved_ids(args.resolved_report)
    stats: dict[str, Any] = {
        "n_traj_files": len(paths),
        "n_resolved_ids": None if resolved is None else len(resolved),
        "only_resolved": resolved is not None,
        "resolved_report": None if resolved is None else str(args.resolved_report),
        "kept_traj": 0,
        "skipped_filter": 0,
        "skipped_unresolved": 0,
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
        if not traj.get("instance_id"):
            traj["instance_id"] = path.parent.name
        iid = str(traj.get("instance_id") or path.parent.name)
        if resolved is not None and iid not in resolved:
            stats["skipped_unresolved"] += 1
            continue
        if not keep_traj(traj, preds, include_failed=args.include_failed):
            stats["skipped_filter"] += 1
            continue
        samples, skipped = convert_one(
            traj,
            keep_turns=args.keep_turns,
            keep_meta=args.keep_meta,
            instance_id_fallback=path.parent.name,
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
        json.dumps(
            {"out": str(args.out), "gen_dir": str(args.gen_dir), **stats},
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(json.dumps({"out": str(args.out), **stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
