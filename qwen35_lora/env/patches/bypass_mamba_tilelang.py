#!/usr/bin/env python3
"""可选 stub：绕过 mamba_ssm 对旧版 tvm-ffi / tilelang 的强依赖。

本任务是 Qwen MoE LoRA SFT；若装了完整 ``mamba_ssm`` 则可 noop。
Megatron hybrid 导入链仍可能 touch ``mamba_ssm``，冲突时用 stub 预加载
Triton layernorm / SSD 子模块。

必须在 ``import megatron.bridge`` 之前调用 ``apply()``。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def apply() -> str:
    if "mamba_ssm" in sys.modules and not getattr(sys.modules["mamba_ssm"], "_fa4_stub", False):
        return "mamba_ssm already imported; bypass skipped"

    spec = importlib.util.find_spec("mamba_ssm")
    if spec is None or not spec.submodule_search_locations:
        return "mamba_ssm not installed; bypass noop"

    root = Path(list(spec.submodule_search_locations)[0])
    pkg = types.ModuleType("mamba_ssm")
    pkg.__path__ = [str(root)]
    pkg.__file__ = str(root / "__init__.py")
    pkg.__package__ = "mamba_ssm"
    pkg.__version__ = "stub-for-fa4"
    pkg._fa4_stub = True
    sys.modules["mamba_ssm"] = pkg

    # 只保留 hybrid import 需要的 Triton ops，避免 Mamba3/tilelang。
    for mod_name, rel in (
        ("mamba_ssm.ops", "ops"),
        ("mamba_ssm.ops.triton", "ops/triton"),
        ("mamba_ssm.ops.triton.layernorm_gated", "ops/triton/layernorm_gated.py"),
        ("mamba_ssm.ops.triton.ssd_combined", "ops/triton/ssd_combined.py"),
    ):
        if mod_name in sys.modules:
            continue
        path = root / rel
        if path.is_dir():
            m = types.ModuleType(mod_name)
            m.__path__ = [str(path)]
            m.__file__ = str(path / "__init__.py")
            m.__package__ = mod_name
            sys.modules[mod_name] = m
            continue
        if not path.is_file():
            continue
        mspec = importlib.util.spec_from_file_location(mod_name, path)
        if mspec is None or mspec.loader is None:
            continue
        module = importlib.util.module_from_spec(mspec)
        sys.modules[mod_name] = module
        try:
            mspec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover
            sys.modules.pop(mod_name, None)
            return f"mamba bypass partial failure on {mod_name}: {exc}"

    print("[bypass_mamba_tilelang] Stubbed mamba_ssm eager imports")
    return "mamba_ssm tilelang bypass applied"
