"""动态加载外部脚本（含中文文件名），不改动原文件。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Optional


def ensure_sys_path(paths: Iterable[Path | str]) -> None:
    for p in paths:
        s = str(Path(p).resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def load_module_from_path(
    module_name: str,
    file_path: Path | str,
    *,
    extra_sys_paths: Optional[Iterable[Path | str]] = None,
) -> ModuleType:
    """按文件路径加载模块；若已加载同名模块则直接返回。"""
    if extra_sys_paths:
        ensure_sys_path(extra_sys_paths)

    file_path = Path(file_path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"模块文件不存在: {file_path}")

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {file_path} 创建 import spec")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
