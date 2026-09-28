#!/usr/bin/env python3
"""把 tau3-banking ``results.json`` 转成 Megatron / Swift chat SFT JSONL。

规则（对齐 ``ycm/doc/Megatron数据格式速记.md``）:
- 只保留成功轨迹（默认 reward >= 1.0）
- 按每个 assistant 轮展开；历史 assistant ``loss=false``，目标轮 ``loss=true``
- 把 ``reasoning_content`` / ``tool_calls`` 渲染进 ``assistant.content``
- ``tools`` 输出为 JSON 字符串
- ``--keep-turns`` 控制哪些 assistant 轮进入训练：
  ``skip_greeting``（默认）/ ``tool_call`` / ``all``

用法::

    python3 /kwkj-k8s/llm_team/ycm/megatron_qwen35/scripts/convert_tau3_results_to_sft_jsonl.py \\
      --results /path/to/results.json \\
      --out /path/to/out.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def _tool_call_name_args(tc: dict[str, Any]) -> tuple[str, Any]:
    """兼容 tau2 扁平 tool_call 与 OpenAI function 嵌套两种形态。"""
    if "function" in tc and isinstance(tc["function"], dict):
        fn = tc["function"]
        name = fn.get("name") or ""
        args = fn.get("arguments", {})
    else:
        name = tc.get("name") or ""
        args = tc.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return name, args


def _format_param_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_tool_calls(tool_calls: list[dict[str, Any]] | None) -> str:
    if not tool_calls:
        return ""
    blocks: list[str] = []
    for tc in tool_calls:
        name, args = _tool_call_name_args(_as_dict(tc))
        if not name:
            continue
        lines = ["<tool_call>", f"<function={name}>"]
        if isinstance(args, dict):
            for k, v in args.items():
                lines.append(f"<parameter={k}>")
                lines.append(_format_param_value(v))
                lines.append("</parameter>")
        elif args not in (None, ""):
            lines.append("<parameter=arguments>")
            lines.append(_format_param_value(args))
            lines.append("</parameter>")
        lines.append("</function>")
        lines.append("</tool_call>")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def extract_reasoning(msg: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning", "thinking"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    raw = msg.get("raw_data")
    if isinstance(raw, dict):
        choices = raw.get("choices") or []
        if choices:
            inner = (choices[0] or {}).get("message") or {}
            for key in ("reasoning_content", "reasoning", "thinking"):
                val = inner.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return ""


def render_assistant_content(msg: dict[str, Any]) -> str:
    reasoning = extract_reasoning(msg)
    text = (msg.get("content") or "").strip()
    tool_part = render_tool_calls(msg.get("tool_calls"))

    parts: list[str] = []
    if reasoning:
        parts.append(f"<think>\n{reasoning}\n</think>")
    if text:
        parts.append(text)
    if tool_part:
        parts.append(tool_part)
    return "\n".join(parts).strip()


def render_user_content(msg: dict[str, Any]) -> str:
    text = (msg.get("content") or "").strip()
    tool_part = render_tool_calls(msg.get("tool_calls"))
    if text and tool_part:
        return f"{text}\n{tool_part}"
    return text or tool_part


def _to_flat_tool(t: dict[str, Any]) -> dict[str, Any] | None:
    """收敛成速记文档的扁 schema：``{name, description, parameters}``。"""
    t = _as_dict(t)
    if t.get("type") == "function" and isinstance(t.get("function"), dict):
        t = t["function"]
    name = t.get("name")
    if not name:
        return None
    return {
        "name": name,
        "description": t.get("description") or "",
        "parameters": t.get("parameters")
        or t.get("input_schema")
        or {"type": "object", "properties": {}},
    }


def normalize_tools(tool_defs: Any, observed: dict[str, set[str]]) -> str | None:
    """优先用 results 里的 tool_defs；否则按轨迹里见过的工具拼最小扁 schema。

    输出对齐 ``Megatron数据格式速记.md``：
    ``[{"name": "...", "description": "...", "parameters": {...}}]`` 的 JSON 字符串。
    """
    tools: list[dict[str, Any]] = []

    if isinstance(tool_defs, str):
        try:
            tool_defs = json.loads(tool_defs)
        except json.JSONDecodeError:
            tool_defs = None

    if isinstance(tool_defs, list) and tool_defs:
        for t in tool_defs:
            flat = _to_flat_tool(_as_dict(t))
            if flat:
                tools.append(flat)
    elif observed:
        for name in sorted(observed):
            props = {k: {"type": "string"} for k in sorted(observed[name])}
            tools.append(
                {
                    "name": name,
                    "description": f"tau3 tool `{name}` (schema inferred from trajectory)",
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": sorted(observed[name]),
                    },
                }
            )

    if not tools:
        return None
    return json.dumps(tools, ensure_ascii=False)


def collect_observed_tools(messages: list[dict[str, Any]]) -> dict[str, set[str]]:
    observed: dict[str, set[str]] = {}
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            name, args = _tool_call_name_args(_as_dict(tc))
            if not name:
                continue
            keys = set(args.keys()) if isinstance(args, dict) else set()
            observed.setdefault(name, set()).update(keys)
    return observed


def convert_messages(
    raw_messages: list[dict[str, Any]],
    *,
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        out.append({"role": "system", "content": system_prompt.strip()})

    for msg in raw_messages:
        role = msg.get("role")
        if role == "assistant":
            content = render_assistant_content(msg)
            if not content:
                continue
            out.append({"role": "assistant", "content": content})
        elif role == "user":
            content = render_user_content(msg)
            if not content:
                continue
            out.append({"role": "user", "content": content})
        elif role == "tool":
            content = msg.get("content")
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            tool_msg: dict[str, Any] = {"role": "tool", "content": content}
            # tau2 tool 消息用 id 关联 tool_call
            tcid = msg.get("tool_call_id") or msg.get("id")
            if tcid:
                tool_msg["tool_call_id"] = tcid
            out.append(tool_msg)
        elif role == "system":
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if out and out[0]["role"] == "system":
                out[0]["content"] = out[0]["content"] + "\n\n" + content
            else:
                out.insert(0, {"role": "system", "content": content})
        else:
            # 跳过未知 role
            continue

    # 合并连续同 role（system/user），避免 chat template 不稳定
    merged: list[dict[str, Any]] = []
    for m in out:
        if (
            merged
            and m["role"] == merged[-1]["role"]
            and m["role"] in {"system", "user"}
            and "tool_call_id" not in m
            and "tool_call_id" not in merged[-1]
        ):
            merged[-1]["content"] = (
                (merged[-1].get("content") or "") + "\n\n" + (m.get("content") or "")
            ).strip()
        else:
            merged.append(m)
    return merged


def assistant_has_tool_call(content: str) -> bool:
    return "<tool_call>" in (content or "")


def should_keep_assistant_turn(
    messages: list[dict[str, Any]],
    target_idx: int,
    *,
    keep_turns: str,
) -> bool:
    """决定某一轮 assistant 要不要展开成训练样本。

    - all: 全部保留
    - tool_call: 只保留 content 里含 ``<tool_call>`` 的轮
    - skip_greeting: 丢掉「任何 user 之前、且无 tool_call」的开场寒暄；
      其余（含最终自然语言答复）都保留
    """
    content = messages[target_idx].get("content") or ""
    has_tool = assistant_has_tool_call(content)

    if keep_turns == "all":
        return True
    if keep_turns == "tool_call":
        return has_tool
    if keep_turns == "skip_greeting":
        if has_tool:
            return True
        # 开场问候：目标轮之前还没有任何 user
        saw_user = any(m["role"] == "user" for m in messages[:target_idx])
        return saw_user
    raise ValueError(f"unknown keep_turns={keep_turns!r}")


def expand_by_assistant_turns(
    messages: list[dict[str, Any]],
    *,
    tools_str: str | None,
    meta: dict[str, Any],
    keep_turns: str = "skip_greeting",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    assistant_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    samples: list[dict[str, Any]] = []
    skipped = {"skipped_no_tool_call": 0, "skipped_greeting": 0}

    for turn_i, target_idx in enumerate(assistant_idxs):
        content = messages[target_idx].get("content") or ""
        has_tool = assistant_has_tool_call(content)
        if not should_keep_assistant_turn(
            messages, target_idx, keep_turns=keep_turns
        ):
            if keep_turns == "tool_call" and not has_tool:
                skipped["skipped_no_tool_call"] += 1
            elif keep_turns == "skip_greeting":
                skipped["skipped_greeting"] += 1
            else:
                skipped["skipped_no_tool_call"] += 1
            continue

        prefix = messages[: target_idx + 1]
        labeled: list[dict[str, Any]] = []
        for i, m in enumerate(prefix):
            item = dict(m)
            if item["role"] == "assistant":
                item["loss"] = i == target_idx
            labeled.append(item)
        row: dict[str, Any] = {"messages": labeled}
        if tools_str:
            row["tools"] = tools_str
        row["meta"] = {
            **meta,
            "assistant_turn": turn_i,
            "n_assistant_turns": len(assistant_idxs),
            "target_has_tool_call": has_tool,
        }
        samples.append(row)
    return samples, skipped


def sim_reward(sim: dict[str, Any]) -> float | None:
    info = sim.get("reward_info") or {}
    if isinstance(info, dict) and "reward" in info:
        try:
            return float(info["reward"])
        except (TypeError, ValueError):
            return None
    return None


def convert_results(
    results: dict[str, Any],
    *,
    min_reward: float,
    include_failed: bool,
    keep_meta: bool,
    keep_turns: str = "skip_greeting",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sims = results.get("simulations") or []
    env = (results.get("info") or {}).get("environment_info") or {}
    default_policy = env.get("policy") or ""
    tool_defs = env.get("tool_defs")

    stats = {
        "n_simulations": len(sims),
        "kept_sims": 0,
        "skipped_reward": 0,
        "skipped_no_assistant": 0,
        "skipped_greeting": 0,
        "skipped_no_tool_call": 0,
        "n_samples": 0,
        "keep_turns": keep_turns,
    }
    rows: list[dict[str, Any]] = []

    for sim in sims:
        reward = sim_reward(sim)
        if not include_failed:
            if reward is None or reward < min_reward:
                stats["skipped_reward"] += 1
                continue

        raw_messages = [_as_dict(m) for m in (sim.get("messages") or [])]
        if not raw_messages:
            stats["skipped_no_assistant"] += 1
            continue

        system_prompt = sim.get("policy") or default_policy
        messages = convert_messages(raw_messages, system_prompt=system_prompt)
        if not any(m["role"] == "assistant" for m in messages):
            stats["skipped_no_assistant"] += 1
            continue

        observed = collect_observed_tools(raw_messages)
        # user 侧工具（如 apply_for_credit_card）一般不进 agent tools；只保留 assistant 调用过的
        agent_observed: dict[str, set[str]] = {}
        for msg in raw_messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                name, args = _tool_call_name_args(_as_dict(tc))
                if not name:
                    continue
                keys = set(args.keys()) if isinstance(args, dict) else set()
                agent_observed.setdefault(name, set()).update(keys)
        tools_str = normalize_tools(tool_defs, agent_observed or observed)

        meta = {
            "task_id": sim.get("task_id"),
            "simulation_id": sim.get("id"),
            "trial": sim.get("trial"),
            "reward": reward,
            "termination_reason": sim.get("termination_reason"),
        }
        samples, skipped = expand_by_assistant_turns(
            messages,
            tools_str=tools_str,
            meta=meta,
            keep_turns=keep_turns,
        )
        stats["skipped_greeting"] += skipped["skipped_greeting"]
        stats["skipped_no_tool_call"] += skipped["skipped_no_tool_call"]
        if not keep_meta:
            for s in samples:
                s.pop("meta", None)
        rows.extend(samples)
        stats["kept_sims"] += 1
        stats["n_samples"] += len(samples)

    return rows, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="tau3 results.json 路径",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--min-reward",
        type=float,
        default=1.0,
        help="保留 reward >= 该阈值的轨迹（默认 1.0）",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="不过滤 reward，失败轨迹也转换",
    )
    parser.add_argument(
        "--keep-meta",
        action="store_true",
        help="在每条样本保留 meta（task_id / reward / turn 等）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最多写多少条样本（0=不限制）",
    )
    parser.add_argument(
        "--keep-turns",
        choices=("all", "tool_call", "skip_greeting"),
        default="skip_greeting",
        help=(
            "assistant 轮过滤：all=全留；"
            "tool_call=只留含 <tool_call> 的轮（无 tool_call 的最终答复会被跳过）；"
            "skip_greeting=丢掉开场寒暄，保留 tool_call 与最终文本答复（默认）"
        ),
    )
    args = parser.parse_args(argv)

    results = json.loads(args.results.read_text())
    rows, stats = convert_results(
        results,
        min_reward=args.min_reward,
        include_failed=args.include_failed,
        keep_meta=args.keep_meta,
        keep_turns=args.keep_turns,
    )
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
        stats["n_samples_written"] = len(rows)
    else:
        stats["n_samples_written"] = len(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"out": str(args.out), **stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
