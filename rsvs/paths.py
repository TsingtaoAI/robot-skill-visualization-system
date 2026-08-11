"""路径全部相对 newtest 自身；不索引仓库外文件，便于只拷贝本文件夹。"""

from __future__ import annotations

from pathlib import Path

NEWTEST_ROOT = Path(__file__).resolve().parent

VENDOR_DIR = NEWTEST_ROOT / "vendor"
ASSETS_DIR = NEWTEST_ROOT / "assets"
WEIGHTS_DIR = NEWTEST_ROOT / "weights"

# ── 内置依赖（全部在 newtest/ 内）────────────────────────
VENDOR_LEGGED_GYM = VENDOR_DIR / "LeggedGym-Ex"
VENDOR_COMBO_SCRIPT = VENDOR_DIR / "skills" / "go2_eval_combo_gym_policy.py"
VENDOR_NAV_DIR = VENDOR_DIR / "nav"
# 与 change/go2_eval_multivel.py 同步的完整导航（Lidar + A* + 排斥 + Viser）
VENDOR_NAV_EVAL = VENDOR_NAV_DIR / "go2_eval_multivel.py"

GO2_MJCF = ASSETS_DIR / "go2" / "go2.xml"  # 导航 / Viser 轻量 STL
GO2_SKILL_MJCF = ASSETS_DIR / "go2_skill" / "go2.xml"  # 技能仿真（与 My_unitree combo 同源）
PLANE_URDF = ASSETS_DIR / "plane_urdf" / "plane.urdf"
GO2_SKILL_URDF = ASSETS_DIR / "go2_urdf" / "go2.urdf"  # 备用（backflip 训练域）

# 优先使用 newtest 根下复制的 legged_gym / rsl_rl / resources（与 LeggedGym-Ex 布局一致）
_TOP_LEGGED = NEWTEST_ROOT / "legged_gym"
_TOP_RSL = NEWTEST_ROOT / "rsl_rl"
if _TOP_LEGGED.is_dir() and _TOP_RSL.is_dir():
    LEGGED_GYM_ROOT = NEWTEST_ROOT
else:
    LEGGED_GYM_ROOT = VENDOR_LEGGED_GYM
# combo 内 resolve_path 的“仓库根”：指向 newtest，使相对路径落在本文件夹
GENESIS_ROOT = NEWTEST_ROOT
LOCOMOTION_DIR = VENDOR_DIR / "skills"
CHANGE_DIR = VENDOR_NAV_DIR
NAV_EVAL_SCRIPT = VENDOR_NAV_EVAL
NAV_ENV_SCRIPT = VENDOR_NAV_DIR / "go2_env_multivel.py"
# 优先 weights/；兼容 vendor/nav/logs 拷贝
MULTIVEL_LOG_DIR = WEIGHTS_DIR / "go2_multivel"
MULTIVEL_LOG_DIR_FALLBACK = VENDOR_NAV_DIR / "logs" / "go2-multivel_morestable"

WTW_EXPERIMENT = "go2_wtw"
WTW_DEFAULT_RUN = "Jul04_10-58-01_wtw_genesis"
WTW_WEIGHT = WEIGHTS_DIR / "go2_wtw" / "model_5000.pt"
WTW_LOG_RUN_DIR = LEGGED_GYM_ROOT / "logs" / WTW_EXPERIMENT / WTW_DEFAULT_RUN

DEFAULT_HANDSTAND = str(WEIGHTS_DIR / "skills" / "handstand.pt")
DEFAULT_LEGSTAND_CYCLE = str(WEIGHTS_DIR / "skills" / "legstand_cycle.pt")
DEFAULT_BACKFLIP = str(WEIGHTS_DIR / "skills" / "backflip_single.pt")
DEFAULT_BACKFLIP_DOUBLE = str(WEIGHTS_DIR / "skills" / "backflip_double.pt")
DEFAULT_SPRING_JUMP = str(WEIGHTS_DIR / "skills" / "spring_jump.pt")

# 兼容旧名（曾指向 monorepo 父目录；现仅保留变量，不再回退到外部）
REPO_ROOT = NEWTEST_ROOT


def setup_runtime_paths() -> None:
    """插入 newtest 根路径（含 legged_gym/rsl_rl）；Genesis 本体来自 conda。"""
    import sys

    ordered = [
        str(LEGGED_GYM_ROOT),
        str(NEWTEST_ROOT),
        str(NEWTEST_ROOT.parent),  # 仅当仍放在 monorepo 时方便 `import newtest`
    ]
    # 兼容旧布局：若仍保留 vendor/LeggedGym-Ex，次优加入 path
    vendor_lg = str(VENDOR_LEGGED_GYM)
    if vendor_lg not in ordered and VENDOR_LEGGED_GYM.is_dir():
        ordered.append(vendor_lg)

    for p in ordered:
        while p in sys.path:
            sys.path.remove(p)
    for p in reversed(ordered):
        if Path(p).is_dir():
            sys.path.insert(0, p)

    # 清掉无效的空壳 genesis（cwd 误加载）
    top = sys.modules.get("genesis")
    if top is not None and not hasattr(top, "init"):
        for name in list(sys.modules):
            if name == "genesis" or name.startswith("genesis."):
                del sys.modules[name]
