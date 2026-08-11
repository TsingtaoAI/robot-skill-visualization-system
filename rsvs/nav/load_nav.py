"""加载内置导航脚本（newtest/vendor/nav），不依赖仓库外 change/。"""

from __future__ import annotations

from types import ModuleType

from newtest.common.import_utils import load_module_from_path
from newtest.paths import CHANGE_DIR, LEGGED_GYM_ROOT, NAV_EVAL_SCRIPT, NAV_ENV_SCRIPT


def load_nav_eval_module() -> ModuleType:
    if not NAV_EVAL_SCRIPT.is_file():
        raise FileNotFoundError(f"缺少内置导航脚本: {NAV_EVAL_SCRIPT}")
    if not NAV_ENV_SCRIPT.is_file():
        raise FileNotFoundError(f"缺少内置导航环境: {NAV_ENV_SCRIPT}")
    return load_module_from_path(
        "newtest_ext_nav_eval",
        NAV_EVAL_SCRIPT,
        extra_sys_paths=[CHANGE_DIR, LEGGED_GYM_ROOT],
    )
