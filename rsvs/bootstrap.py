"""保证以脚本或 -m 方式启动时都能找到 newtest / vendor。"""

from __future__ import annotations

import os
from pathlib import Path

# 在 import genesis 之前尽量指定 Quadrants 缓存目录（与 run.sh 一致）
_ROOT = Path(__file__).resolve().parent
_QD = _ROOT / ".cache" / "qdcache"
if "QD_OFFLINE_CACHE_FILE_PATH" not in os.environ:
    try:
        (_QD / "python_side_cache").mkdir(parents=True, exist_ok=True)
        (_QD / "kernel_compilation_manager").mkdir(parents=True, exist_ok=True)
        os.environ["QD_OFFLINE_CACHE_FILE_PATH"] = str(_QD)
    except Exception:
        pass

from newtest.paths import setup_runtime_paths

setup_runtime_paths()
