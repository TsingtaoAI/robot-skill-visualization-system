"""

cd ~/genesis/change
python go2_eval_multivel.py -e go2-multivel_morestable --log_dir logs/go2-multivel_morestable --nav_demo  --nav_speed 0.65 






# Lidar 绕障演示（官方 Lidar + A*，输出 nav_demo_*.mp4）
python go2_eval_multivel.py -e go2-multivel_morestable --log_dir logs/go2-multivel_morestable --nav_demo

python go2_eval_multivel.py -e go2-multivel_morestable --log_dir logs/go2-multivel_morestable --demo
go2_eval_multivel.py
====================
多速度策略评估 + 录像。

亮点：支持"速度序列演示"模式（--demo），自动切换不同速度指令，
展示机器人全向行走能力，适合求职作品集视频。

用法：
  # 普通录像（随机速度）
  python go2_eval_multivel.py -e go2-multivel

  # 指定 checkpoint
  python go2_eval_multivel.py -e go2-multivel --ckpt 5000

  # 演示模式：自动展示前进→转向→后退→侧移→原地转
  python go2_eval_multivel.py -e go2-multivel_morestable --log_dir logs/go2-multivel_morestable --demo

  # 演示模式 + 自定义每段时长
  python go2_eval_multivel.py -e go2-multivel --demo --seg_frames 150

  # Lidar 绕障：corridor 场景，依次跑 3 组终点（后墙夹角 / 左后墙 / 右后墙外侧）
  python go2_eval_multivel.py -e go2-multivel_morestable --log_dir logs/go2-multivel_morestable --nav_demo

  # 只跑其中一组终点
  python go2_eval_multivel.py -e go2-multivel_morestable --log_dir logs/go2-multivel_morestable --nav_demo --nav_goal corner

  # Viser 网页交互导航：下拉选场景 + 调速度 + 点击/输入终点
  # 浏览器打开 http://localhost:8080
  python go2_eval_multivel.py -e go2-multivel_morestable --log_dir logs/go2-multivel_morestable \\
      --nav_demo --viewer viser --nav_scene corridor

  # 场景可在网页「场景选择」切换（corridor / fence / open8），也可命令行指定
  python go2_eval_multivel.py -e go2-multivel_morestable --log_dir logs/go2-multivel_morestable \\
      --nav_demo --viewer viser --nav_scene fence --nav_speed 0.65

  # 使用自定义日志目录（如 go2-multivel_morestable）
  python go2_eval_multivel.py -e go2-multivel_morestable --log_dir logs/go2-multivel_morestable

  # 使用自定义日志目录 + 指定 checkpoint
  python go2_eval_multivel.py -e go2-multivel_morestable --log_dir logs/go2-multivel_morestable --ckpt 5000

复制到 Windows：
  cp logs/go2-multivel/eval_*.mp4 /mnt/d/RL/Genesis/
"""

import argparse
import heapq
import math
import os
import pickle
import threading
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name",  type=str, nargs="+", required=True)
    parser.add_argument("--log_dir",         type=str,   default=None,
                        help="自定义日志目录路径，默认使用 logs/{exp_name}")
    parser.add_argument("--ckpt",            type=int,   default=None,
                        help="指定 checkpoint 编号，默认取最新")
    parser.add_argument("--frames",          type=int,   default=600,
                        help="普通模式录制帧数（默认600=12s@50fps）")
    parser.add_argument("--demo",            action="store_true", default=False,
                        help="演示模式：按预设序列切换速度，展示全向能力")
    parser.add_argument("--nav_demo",        action="store_true", default=False,
                        help="Lidar 绕障演示：官方 Lidar 建图 + A*，绕过柱障碍到达目标")
    parser.add_argument("--nav_scene",       type=str, default="corridor",
                        choices=("corridor", "fence", "open8"),
                        help="绕障场景：corridor / fence / open8（Viser 交互模式生效）")
    parser.add_argument("--nav_goal",        type=str, default=None,
                        choices=("corner", "left_rear", "right_rear"),
                        help="corridor 终点变体；非 Viser 且不指定则依次跑全部 3 组")
    parser.add_argument("--seg_frames",      type=int,   default=200,
                        help="演示模式每段帧数（默认200=4s@50fps）")
    parser.add_argument("--nav_frames",      type=int,   default=1500,
                        help="绕障演示最多录制帧数（默认1500=30s@50fps）")
    parser.add_argument("--nav_speed",       type=float, default=0.55,
                        help="绕障路径跟踪速度指令上限（默认0.55m/s）")
    parser.add_argument("--viewer",          type=str, default="native",
                        choices=("native", "viser"),
                        help="可视化：native=离线录制；viser=网页交互点选终点")
    parser.add_argument("--viser_port",      type=int, default=8080,
                        help="Viser 网页端口（--viewer viser）")
    parser.add_argument("--cpu",             action="store_true", default=False)
    return parser.parse_args()


# ── 演示速度序列（面试作品集用）─────────────────────────────────────
# 每个元素：(vx, vy, ang_vel, 描述)
DEMO_SEQUENCE = [
    # ( 0.0,  0.0,  0.0, "静止站立"),
    #( 0.4,  0.4,  0.0, "前进 1.8 m/s"),
     ( 0.0,  0.0,  0.8, "原地左转"),
#     ( 0.8,  0.0,  0.0, "前进 1.8 m/s"),
#     ( 0.0,  0.0, -0.8, "原地右转"),
#     ( 0.0,  0.4,  0.0, "左侧移"),
#     ( 0.0, -0.4,  0.0, "右侧移"),
#     (-1.5,  0.0,  0.0, "后退 1.5 m/s"),
#     ( 0.6,  0.0,  0.5, "前进+左转"),
#     ( 0.0,  0.0,  0.0, "停止"),
]


# ── Lidar 绕障：共享导航参数（所有场景相同）────────────────────────
ROBOT_WIDTH = 0.36
NAV_OBSTACLE_H = 0.55
NAV_PILLAR_SIZE = 0.30
NAV_WALL_THICK = 0.12
NAV_LIDAR_RANGE = 4.5
NAV_LIDAR_SCAN_Z = 0.10
NAV_LIDAR_BODY_OFFSET = (0.3, 0.0)
NAV_LIDAR_RAYS = 181
NAV_LIDAR_AZIMUTHS = tuple(
    math.radians(-180.0 + 359.0 * i / (NAV_LIDAR_RAYS - 1))
    for i in range(NAV_LIDAR_RAYS)
)
NAV_GRID_RESOLUTION = 0.05
NAV_OBSTACLE_INFLATION = 0.12
NAV_MAP_INFLATION = 0.12
NAV_RAY_STOP_MARGIN = 0.10
NAV_ROBOT_CLEAR = 0.22
NAV_REPLAN_INTERVAL = 5
NAV_LOOKAHEAD = 0.65
NAV_GOAL_REACH_DIST = 0.35
NAV_HIT_REPULSE_DIST = 0.38
NAV_HIT_REPULSE_GAIN = 0.25
NAV_CAMERA_FOLLOW = (None, None, 2.2)

# 规划地图相对场景几何外扩；可选终点区相对规划地图内缩（避免贴边导致路径穿出）
NAV_PLAN_EXPAND = 0.80   # m，真实规划世界每侧扩大
NAV_GOAL_INSET = 0.40    # m，可选终点方块比规划世界更小

# ── 场景几何（仅改此处切换布局，不动上方导航参数）────────────────────
_OBST_H = NAV_OBSTACLE_H
_WT = NAV_WALL_THICK
_PS = NAV_PILLAR_SIZE


def _pad_bounds(bounds, pad):
    xmin, xmax, ymin, ymax = bounds
    return (xmin - pad, xmax + pad, ymin - pad, ymax + pad)


def _inset_bounds(bounds, inset):
    xmin, xmax, ymin, ymax = bounds
    return (xmin + inset, xmax - inset, ymin + inset, ymax - inset)


def _derive_plan_and_goal_bounds(core_bounds):
    """core 几何范围 → (扩大后的规划地图, 略小的可选终点区)。"""
    plan = _pad_bounds(core_bounds, NAV_PLAN_EXPAND)
    goal = _inset_bounds(plan, NAV_GOAL_INSET)
    return plan, goal


NAV_SCENES = {
    # 内嵌通道 + 7 柱（当前默认验证场景）
    "corridor": {
        "label": "内嵌通道 + 7柱",
        "arena_size": 6.0,
        "start": (1.15, 1.15),
        "goal": (5.5, 5.5),
        # core_bounds：障碍/几何占用的核心范围；规划会再扩大、终点区再内缩
        "core_bounds": (-0.35, 6.35, -0.35, 6.35),
        "walls": (
            {"pos": (0.65, 3.00), "size": (_WT, 4.70, _OBST_H)},
            {"pos": (3.00, 0.65), "size": (4.70, _WT, _OBST_H)},
            {"pos": (5.35, 2.00), "size": (_WT, 2.80, _OBST_H)},
            {"pos": (2.00, 5.35), "size": (2.80, _WT, _OBST_H)},
        ),
        "pillars": (
            (2.00, 1.60), (1.40, 2.50),
            (2.80, 2.30), (2.20, 3.30),
            (3.60, 3.20), (3.00, 4.20),
            (4.60, 4.40),
        ),
    },
}

# corridor 三组终点（场景几何不变，仅改 goal）
# 机器狗起点 (1.15, 1.15) 朝目标；后方 L 角由西墙 x=0.65 与南墙 y=0.65 构成
_WALL_OUTER = 0.65 - _WT * 0.5          # 墙体外侧面 ≈ 0.59
_GOAL_BEHIND_MARGIN = 0.35              # 墙外再留一段，便于到达判定
_BEHIND_WALL = _WALL_OUTER - _GOAL_BEHIND_MARGIN  # ≈ 0.24

NAV_CORRIDOR_GOALS = (
    {
        "id": "corner",
        "label": "后墙夹角外侧（墙后面）",
        "goal": (_BEHIND_WALL, _BEHIND_WALL),
    },
    {
        "id": "left_rear",
        "label": "左后墙（西墙）中心外侧",
        "goal": (_BEHIND_WALL, 3.00),
    },
    {
        "id": "right_rear",
        "label": "右后墙（南墙）中心外侧",
        "goal": (3.00, _BEHIND_WALL),
    },
)

# 保留 fence / open8 定义供参考，nav_demo 不再自动运行
_NAV_SCENES_EXTRA = {
    # 小围栏：狗在笼内朝 +X 前墙，-X 后方留出口，目标在前墙外侧
    "fence": {
        "label": "小围栏出圈（后开口→绕至前墙外）",
        "arena_size": 6.0,
        "start": (0.0, 0.0),
        "goal": (1.20, 0.0),
        "core_bounds": (-1.50, 2.20, -1.20, 1.20),
        "walls": (
            {"pos": (0.65, 0.00), "size": (_WT, 1.48, _OBST_H)},
            {"pos": (-0.85, 0.49), "size": (_WT, 0.26, _OBST_H)},
            {"pos": (-0.85, -0.49), "size": (_WT, 0.26, _OBST_H)},
            {"pos": (-0.10, 0.62), "size": (1.74, _WT, _OBST_H)},
            {"pos": (-0.10, -0.62), "size": (1.74, _WT, _OBST_H)},
        ),
        "pillars": (),
    },
    "open8": {
        "label": "开阔地 8柱（对角线散布）",
        "arena_size": 6.0,
        "start": (0.5, 0.5),
        "goal": (5.5, 5.5),
        "core_bounds": (-0.35, 6.35, -0.35, 6.35),
        "walls": (),
        "pillars": (
            (1.6, 1.9), (2.1, 1.4), (2.5, 2.5), (3.0, 1.8),
            (3.5, 3.8), (4.0, 3.0), (4.5, 4.5), (5.0, 4.0),
        ),
    },
}
NAV_SCENES.update(_NAV_SCENES_EXTRA)

# 运行时由 _apply_nav_scene() 写入
NAV_ARENA_SIZE = 6.0
NAV_START_POINT = (1.15, 1.15)
NAV_GOAL_POINT = (5.5, 5.5)
NAV_WALL_SPECS = ()
NAV_PILLAR_POSITIONS = ()
NAV_GRID_BOUNDS = (-0.35, 6.35, -0.35, 6.35)   # 扩大后的规划地图
NAV_GOAL_BOUNDS = (-0.35, 6.35, -0.35, 6.35)   # 略小的可选终点区
NAV_SCENE_NAME = "corridor"


def _corridor_goal_cfg(goal_id):
    for item in NAV_CORRIDOR_GOALS:
        if item["id"] == goal_id:
            return item
    raise ValueError(f"未知 corridor 终点: {goal_id}")


def _apply_nav_scene(scene_name, goal_id=None):
    """加载场景几何到模块级变量（不改导航参数）。"""
    global NAV_ARENA_SIZE, NAV_START_POINT, NAV_GOAL_POINT
    global NAV_WALL_SPECS, NAV_PILLAR_POSITIONS
    global NAV_GRID_BOUNDS, NAV_GOAL_BOUNDS, NAV_SCENE_NAME
    cfg = NAV_SCENES[scene_name]
    NAV_SCENE_NAME = scene_name
    NAV_ARENA_SIZE = cfg["arena_size"]
    NAV_START_POINT = cfg["start"]
    if scene_name == "corridor" and goal_id is not None:
        NAV_GOAL_POINT = _corridor_goal_cfg(goal_id)["goal"]
    else:
        NAV_GOAL_POINT = cfg["goal"]
    NAV_WALL_SPECS = cfg["walls"]
    NAV_PILLAR_POSITIONS = cfg["pillars"]
    core = cfg.get("core_bounds") or cfg.get("grid_bounds")
    plan, goal = _derive_plan_and_goal_bounds(core)
    NAV_GRID_BOUNDS = plan
    NAV_GOAL_BOUNDS = goal
    print(f"  [nav] 规划地图={NAV_GRID_BOUNDS}  可选终点区={NAV_GOAL_BOUNDS}", flush=True)


def _set_nav_goal_xy(xy, robot_xy=None):
    """运行时更新导航终点（世界系 xy），自动投影到可选终点区内。"""
    global NAV_GOAL_POINT
    projected, _, _ = _project_goal_into_world(xy, robot_xy=robot_xy)
    NAV_GOAL_POINT = projected
    return projected


def _rect_bounds(rect, margin=0.0):
    xmin, xmax, ymin, ymax = rect
    return (
        float(xmin) + margin,
        float(xmax) - margin,
        float(ymin) + margin,
        float(ymax) - margin,
    )


def _point_in_rect(xy, rect, margin=0.0):
    xmin, xmax, ymin, ymax = _rect_bounds(rect, margin)
    return xmin <= float(xy[0]) <= xmax and ymin <= float(xy[1]) <= ymax


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _clamp_point_to_rect(xy, rect, margin=0.0):
    xmin, xmax, ymin, ymax = _rect_bounds(rect, margin)
    if xmin > xmax or ymin > ymax:
        xmin, xmax, ymin, ymax = rect
    return (
        _clamp(float(xy[0]), xmin, xmax),
        _clamp(float(xy[1]), ymin, ymax),
    )


def _ray_exit_on_rect(origin_xy, goal_xy, rect, margin=0.0):
    """若起点在矩形内、终点在外：取 origin→goal 与矩形边界的首次离开点。"""
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    xmin, xmax, ymin, ymax = _rect_bounds(rect, margin)
    if xmin > xmax or ymin > ymax:
        xmin, xmax, ymin, ymax = rect

    if not (xmin <= ox <= xmax and ymin <= oy <= ymax):
        return None
    if xmin <= gx <= xmax and ymin <= gy <= ymax:
        return (gx, gy)

    dx, dy = gx - ox, gy - oy
    hits = []
    if abs(dx) > 1e-12:
        for x_edge in (xmin, xmax):
            t = (x_edge - ox) / dx
            if 0.0 < t <= 1.0:
                y = oy + t * dy
                if ymin - 1e-8 <= y <= ymax + 1e-8:
                    hits.append(t)
    if abs(dy) > 1e-12:
        for y_edge in (ymin, ymax):
            t = (y_edge - oy) / dy
            if 0.0 < t <= 1.0:
                x = ox + t * dx
                if xmin - 1e-8 <= x <= xmax + 1e-8:
                    hits.append(t)
    if not hits:
        return None
    t = min(hits)
    return (ox + t * dx, oy + t * dy)


def _project_goal_into_world(goal_xy, robot_xy=None, margin=0.05):
    """
    将终点约束到「可选终点区」NAV_GOAL_BOUNDS 内：
      1) 已在终点区内 → 原样
      2) 机器人在规划地图内、终点在外 → 连线与终点区边界交点
      3) 否则 → 终点区内最近点
    返回 (final_xy, was_clamped, raw_xy)
    """
    raw = (float(goal_xy[0]), float(goal_xy[1]))
    if _point_in_rect(raw, NAV_GOAL_BOUNDS, margin=margin):
        return raw, False, raw

    projected = None
    if robot_xy is not None and _point_in_rect(robot_xy, NAV_GRID_BOUNDS, margin=0.0):
        projected = _ray_exit_on_rect(robot_xy, raw, NAV_GOAL_BOUNDS, margin=margin)

    if projected is None:
        projected = _clamp_point_to_rect(raw, NAV_GOAL_BOUNDS, margin=margin)

    projected = _clamp_point_to_rect(projected, NAV_GOAL_BOUNDS, margin=margin)
    return projected, True, raw


def _ray_hit_ground(origin, direction, z_plane=0.0):
    """射线与水平面 z=z_plane 求交，返回 (x, y) 或 None。"""
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
    if abs(dz) < 1e-8:
        return None
    t = (z_plane - oz) / dz
    if t < 0.0:
        return None
    return (ox + t * dx, oy + t * dy)


def _find_go2_mjcf():
    """定位 Go2 MJCF（含 STL 网格），优先 newtest/assets 与内置 LeggedGym-Ex。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "assets", "go2", "go2.xml"),
        os.path.join(
            here, "..", "LeggedGym-Ex", "resources",
            "robots", "unitree_robotics", "go2", "go2.xml",
        ),
        os.path.join(
            here, "..", "LeggedGym-Ex_original", "resources",
            "robots", "unitree_robotics", "go2", "go2.xml",
        ),
        os.path.expanduser(
            "~/genesis/LeggedGym-Ex/resources/robots/unitree_robotics/go2/go2.xml"
        ),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(path):
            return path
    return None


def _rpy_to_matrix(rpy):
    import numpy as np
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _load_go2_body_meshes():
    """从 go2.xml 加载各 body 的视觉网格（连杆局部系）。失败返回空 dict。"""
    import xml.etree.ElementTree as ET
    import numpy as np
    try:
        import trimesh
    except ImportError:
        print("  [viser] 未安装 trimesh，机器人将显示为方块占位")
        return {}

    xml_path = _find_go2_mjcf()
    if xml_path is None:
        print("  [viser] 未找到 go2.xml，机器人将显示为方块占位")
        return {}

    root = ET.parse(xml_path).getroot()
    xml_dir = os.path.dirname(xml_path)
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "meshes") if compiler is not None else "meshes"
    mesh_files = {}
    asset = root.find("asset")
    if asset is not None:
        for mesh in asset.findall("mesh"):
            name = mesh.get("name")
            fname = mesh.get("file")
            if name and fname:
                mesh_files[name] = os.path.join(xml_dir, meshdir, fname)

    materials = {}
    if asset is not None:
        for mat in asset.findall("material"):
            name = mat.get("name")
            rgba = mat.get("rgba")
            if name and rgba:
                materials[name] = np.array([float(x) for x in rgba.split()], dtype=np.float64)

    def parse_xyz(s, default=(0.0, 0.0, 0.0)):
        if not s:
            return np.array(default, dtype=np.float64)
        return np.array([float(x) for x in s.split()], dtype=np.float64)

    body_meshes = {}
    cache = {}

    def walk(body_elem):
        body_name = body_elem.get("name")
        parts = []
        for geom in body_elem.findall("geom"):
            mesh_name = geom.get("mesh")
            if not mesh_name or mesh_name not in mesh_files:
                continue
            class_name = geom.get("class", "")
            if class_name and "collision" in class_name:
                continue
            full = mesh_files[mesh_name]
            if not os.path.isfile(full):
                continue
            if full in cache:
                mesh = cache[full].copy()
            else:
                try:
                    mesh = trimesh.load(full, force="mesh")
                    cache[full] = mesh.copy()
                except Exception:
                    continue
            T = np.eye(4)
            T[:3, 3] = parse_xyz(geom.get("pos"))
            euler = geom.get("euler")
            if euler:
                T[:3, :3] = _rpy_to_matrix(parse_xyz(euler))
            mesh.apply_transform(T)
            mat_name = geom.get("material")
            rgba = materials.get(mat_name, np.array([0.75, 0.78, 0.82, 1.0]))
            color = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
            if color.shape[0] == 3:
                color = np.append(color, 255)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(color, (len(mesh.vertices), 1))
            )
            parts.append(mesh)
        if body_name and parts:
            body_meshes[body_name] = trimesh.util.concatenate(parts)
        for child in body_elem.findall("body"):
            walk(child)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in worldbody.findall("body"):
            walk(body)

    print(f"  [viser] 已加载 Go2 网格 {len(body_meshes)} 个连杆（{xml_path}）")
    return body_meshes


class NavViserUI:
    """Viser 网页交互：点击地面 / 输入坐标设置导航终点，并高亮显示。"""

    SCENE_KEYS = ("corridor", "fence", "open8")

    def __init__(self, port=8080, scene_name="corridor", nav_speed=0.55):
        try:
            import viser
        except ImportError as exc:
            raise ImportError(
                "需要安装 viser：pip install viser（genesisEx 环境通常已有）"
            ) from exc

        self._viser = viser
        self.server = viser.ViserServer(port=port)
        try:
            from newtest.common.viser_lifecycle import attach_exit_when_browser_closed

            attach_exit_when_browser_closed(self.server, grace_sec=8.0, label="nav")
        except Exception as e:
            print(f"  [viser] 未能挂载关页退出: {e}", flush=True)
        self._lock = threading.Lock()
        self.goal_xy = None
        self.navigating = False
        self.goal_revision = 0
        self.stop_requested = False
        self.restart_requested = False
        self.reset_requested = False
        self.pending_scene = None
        self.nav_speed = float(nav_speed)
        self._status = "请点击地面或输入坐标设置终点，再按「开始导航」"
        self._path_line = None
        self._robot_handle = None          # 方块占位（网格失败时）
        self._body_handles = {}            # link_name -> mesh handle
        self._link_name_to_idx = None      # list[str] 对齐 get_links_pos
        self._goal_handle = None
        self._goal_halo = None
        self._goal_arrow = None
        self._obstacle_handles = []
        self._robot_xy = NAV_SCENES[scene_name]["start"]

        self._build_scene(scene_name)
        self._build_gui(scene_name)
        self._bind_click()

        host = self.server.get_host()
        print(f"  [viser] 网页交互已启动：http://{host}:{port}")
        print("  [viser] 操作：选场景 / 调速度 / 点击或输入终点 →「开始导航」")

    def _build_scene(self, scene_name):
        cfg = NAV_SCENES[scene_name]
        # 地面按扩大后的规划地图；可选终点区画一圈浅色框提示
        xmin, xmax, ymin, ymax = NAV_GRID_BOUNDS
        gx0, gx1, gy0, gy1 = NAV_GOAL_BOUNDS
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        span = max(xmax - xmin, ymax - ymin)

        self.server.scene.add_grid(
            "/ground",
            width=span + 1.0,
            height=span + 1.0,
            position=(cx, cy, 0.0),
        )
        # 可选终点区（比规划地图略小）
        self.server.scene.add_box(
            "/goal_region",
            color=(80, 200, 120),
            dimensions=(max(0.2, gx1 - gx0), max(0.2, gy1 - gy0), 0.01),
            position=(0.5 * (gx0 + gx1), 0.5 * (gy0 + gy1), 0.005),
            opacity=0.12,
            wireframe=True,
        )

        z = NAV_OBSTACLE_H * 0.5
        self._obstacle_handles = []
        for i, wall in enumerate(cfg["walls"]):
            self._obstacle_handles.append(
                self.server.scene.add_box(
                    f"/obstacles/wall_{i}",
                    color=(120, 120, 130),
                    dimensions=tuple(wall["size"]),
                    position=(wall["pos"][0], wall["pos"][1], z),
                )
            )
        for i, (px, py) in enumerate(cfg["pillars"]):
            self._obstacle_handles.append(
                self.server.scene.add_box(
                    f"/obstacles/pillar_{i}",
                    color=(90, 90, 100),
                    dimensions=(NAV_PILLAR_SIZE, NAV_PILLAR_SIZE, NAV_OBSTACLE_H),
                    position=(px, py, z),
                )
            )

        sx, sy = cfg["start"]
        self.server.scene.add_icosphere(
            "/start_marker",
            radius=0.08,
            color=(80, 180, 255),
            position=(sx, sy, 0.08),
        )
        self._add_robot_visuals(sx, sy)

        # 终点高亮：外圈光晕 + 实心球 + 竖向引导
        self._goal_halo = self.server.scene.add_icosphere(
            "/goal_halo",
            radius=0.22,
            color=(255, 220, 40),
            opacity=0.35,
            position=(0.0, 0.0, 0.10),
            visible=False,
        )
        self._goal_handle = self.server.scene.add_icosphere(
            "/goal_marker",
            radius=0.10,
            color=(40, 255, 80),
            position=(0.0, 0.0, 0.12),
            visible=False,
        )
        self._goal_arrow = self.server.scene.add_box(
            "/goal_stem",
            color=(40, 255, 80),
            dimensions=(0.04, 0.04, 0.35),
            position=(0.0, 0.0, 0.28),
            visible=False,
        )

        @self.server.on_client_connect
        def _(client):
            client.camera.position = (sx - 2.2, sy - 3.0, 3.2)
            client.camera.look_at = (sx + 1.5, sy + 1.5, 0.2)

    def _add_robot_visuals(self, sx, sy):
        """优先加载 Go2 STL 网格；失败则退回蓝色方块占位。"""
        body_meshes = _load_go2_body_meshes()
        self._body_handles = {}
        self._robot_handle = None
        if body_meshes:
            for body_name, mesh in body_meshes.items():
                handle = self.server.scene.add_mesh_trimesh(
                    f"/robot/{body_name}",
                    mesh,
                    cast_shadow=True,
                    receive_shadow=True,
                )
                # 初始放到起点附近；首帧 update_robot_from_env 会纠正
                handle.position = (sx, sy, 0.35)
                self._body_handles[body_name] = handle
            return

        self._robot_handle = self.server.scene.add_box(
            "/robot",
            color=(40, 160, 255),
            dimensions=(0.35, 0.22, 0.28),
            position=(sx, sy, 0.20),
        )

    @staticmethod
    def _scene_option_label(key):
        return f"{key} — {NAV_SCENES[key]['label']}"

    @staticmethod
    def _scene_key_from_option(option):
        return option.split(" — ", 1)[0].strip()

    def _build_gui(self, scene_name):
        cfg = NAV_SCENES[scene_name]
        gx, gy = cfg["goal"]
        xmin, xmax, ymin, ymax = NAV_GOAL_BOUNDS

        self._status_md = self.server.gui.add_markdown(
            f"**场景**：{scene_name}（{cfg['label']}）\n\n{self._status}"
        )
        self._scene_label = scene_name
        self.reset_requested = False

        with self.server.gui.add_folder("场景选择", expand_by_default=True):
            scene_opts = tuple(self._scene_option_label(k) for k in self.SCENE_KEYS)
            init_opt = self._scene_option_label(scene_name)
            self._scene_dd = self.server.gui.add_dropdown(
                "场景",
                options=scene_opts,
                initial_value=init_opt,
                hint="corridor / fence / open8（与 _3场景 参考一致）",
            )
            btn_load_scene = self.server.gui.add_button(
                "加载场景",
                color="violet",
                hint="切换场景需重建仿真（会短暂重启进程）",
            )
            btn_reset = self.server.gui.add_button(
                "重置",
                color="orange",
                hint="机器狗回到起点，清空终点与地图记忆",
            )

            @btn_load_scene.on_click
            def _(_e):
                key = self._scene_key_from_option(self._scene_dd.value)
                if key not in NAV_SCENES:
                    self._set_status(f"未知场景：{self._scene_dd.value}")
                    return
                if key == self._scene_label:
                    self._set_status(f"当前已是场景 {key}，无需重新加载")
                    return
                with self._lock:
                    self.pending_scene = key
                    self.restart_requested = True
                    self.navigating = False
                self._set_status(
                    f"正在切换到 {key}（{NAV_SCENES[key]['label']}），请稍候…"
                )

            @btn_reset.on_click
            def _(_e):
                with self._lock:
                    self.reset_requested = True
                    self.navigating = False
                    self.stop_requested = True
                    self.goal_xy = None
                    self.goal_revision += 1
                self._goal_handle.visible = False
                self._goal_halo.visible = False
                self._goal_arrow.visible = False
                if self._path_line is not None:
                    try:
                        self._path_line.remove()
                    except Exception:
                        pass
                    self._path_line = None
                self._set_status("已请求重置：机器狗将回到起点")

        with self.server.gui.add_folder("速度控制", expand_by_default=True):
            self._speed_num = self.server.gui.add_number(
                "导航速度 (m/s)",
                initial_value=float(self.nav_speed),
                min=0.10,
                max=1.00,
                step=0.05,
                hint="路径跟踪指令上限；切换场景会保留。转向大时会暂时低于此值",
            )
            self._speed_slider = self.server.gui.add_slider(
                "速度滑条",
                min=0.10,
                max=1.00,
                step=0.05,
                initial_value=float(self.nav_speed),
            )
            self.server.gui.add_markdown(
                f"当前载入设定：**{self.nav_speed:.2f} m/s**"
                f"（训练侧移上限 0.5，正对目标时向前可达约 1.0）"
            )

            @self._speed_num.on_update
            def _(_e):
                v = float(self._speed_num.value)
                self.nav_speed = v
                if abs(float(self._speed_slider.value) - v) > 1e-6:
                    self._speed_slider.value = v
                self._set_status(f"导航速度已设为 {v:.2f} m/s")

            @self._speed_slider.on_update
            def _(_e):
                v = float(self._speed_slider.value)
                self.nav_speed = v
                if abs(float(self._speed_num.value) - v) > 1e-6:
                    self._speed_num.value = v

        with self.server.gui.add_folder("导航终点", expand_by_default=True):
            self._goal_vec = self.server.gui.add_vector2(
                "坐标 (x, y)",
                initial_value=(float(gx), float(gy)),
                min=(float(xmin), float(ymin)),
                max=(float(xmax), float(ymax)),
                step=0.05,
                hint="输入后点「预览终点」显示高亮；或直接点击 3D 场景地面",
            )
            btn_preview = self.server.gui.add_button("预览终点", color="cyan")
            btn_go = self.server.gui.add_button("开始导航", color="green")
            btn_stop = self.server.gui.add_button("停止", color="red")
            btn_clear = self.server.gui.add_button("清除终点")

            if scene_name == "corridor":
                presets = ["(自定义)"] + [g["id"] for g in NAV_CORRIDOR_GOALS]
                self._preset = self.server.gui.add_dropdown(
                    "corridor 预设终点",
                    options=tuple(presets),
                    initial_value="(自定义)",
                )

                @self._preset.on_update
                def _(_e):
                    if self._preset.value == "(自定义)":
                        return
                    g = _corridor_goal_cfg(self._preset.value)["goal"]
                    self._goal_vec.value = (float(g[0]), float(g[1]))
                    self.preview_goal(g, source=f"预设:{self._preset.value}")

            @btn_preview.on_click
            def _(_e):
                self.preview_goal(self._goal_vec.value, source="输入坐标")

            @btn_go.on_click
            def _(_e):
                self.start_navigation(self._goal_vec.value)

            @btn_stop.on_click
            def _(_e):
                with self._lock:
                    self.navigating = False
                    self.stop_requested = True
                self._set_status("已停止，可重新设置终点")

            @btn_clear.on_click
            def _(_e):
                with self._lock:
                    self.goal_xy = None
                    self.navigating = False
                    self.stop_requested = True
                    self.goal_revision += 1
                self._goal_handle.visible = False
                self._goal_halo.visible = False
                self._goal_arrow.visible = False
                self._set_status("终点已清除")

        with self.server.gui.add_folder("说明"):
            self.server.gui.add_markdown(
                "- **场景选择**：下拉选通道 / 围栏 / 开阔地，再点「加载场景」\n"
                "- **重置**：机器狗回起点，清空终点与建图\n"
                "- **速度控制**：输入框或滑条改导航速度 (m/s)\n"
                "- **绿线方框**：可选终点区（比规划地图略小，防贴边）\n"
                "- **点击地面** / **输入坐标**：设置终点并高亮（区外自动投影）\n"
                "-「开始导航」：机器狗开始前往当前终点"
            )

    def _bind_click(self):
        @self.server.scene.on_click()
        def _(event):
            hit = _ray_hit_ground(event.ray_origin, event.ray_direction, z_plane=0.0)
            if hit is None:
                return
            # 越界会在 preview_goal 内自动投影到地图内
            self.preview_goal(hit, source="点击地面", robot_xy=self._robot_xy)

    def _set_status(self, text):
        self._status = text
        label = NAV_SCENES.get(self._scene_label, {}).get("label", self._scene_label)
        goal = self.goal_xy
        goal_txt = "无" if goal is None else f"({goal[0]:.2f}, {goal[1]:.2f})"
        nav = "导航中" if self.navigating else "待机"
        self._status_md.content = (
            f"**场景**：{self._scene_label}（{label}）  |  **状态**：{nav}  |  "
            f"**速度**：{self.nav_speed:.2f} m/s\n\n"
            f"**当前终点**：{goal_txt}\n\n{text}"
        )

    def preview_goal(self, xy, source="坐标", robot_xy=None):
        raw = (float(xy[0]), float(xy[1]))
        ref = robot_xy if robot_xy is not None else self._robot_xy
        projected, clamped, raw_xy = _project_goal_into_world(raw, robot_xy=ref)
        x, y = projected
        with self._lock:
            self.goal_xy = (x, y)
            self.goal_revision += 1
        try:
            self._goal_vec.value = (float(x), float(y))
        except Exception:
            pass
        pos = (x, y, 0.12)
        self._goal_handle.position = pos
        self._goal_handle.visible = True
        self._goal_halo.position = (x, y, 0.08)
        self._goal_halo.visible = True
        self._goal_arrow.position = (x, y, 0.28)
        self._goal_arrow.visible = True
        if clamped:
            self._set_status(
                f"终点在地图外，已投影到边界内："
                f"({raw_xy[0]:.2f},{raw_xy[1]:.2f}) → ({x:.2f},{y:.2f})，"
                f"按「开始导航」出发"
            )
        else:
            self._set_status(
                f"已设置终点（{source}）：({x:.2f}, {y:.2f})，按「开始导航」出发"
            )
        return projected, clamped

    def start_navigation(self, xy=None):
        if xy is not None:
            self.preview_goal(xy, source="开始导航", robot_xy=self._robot_xy)
        with self._lock:
            if self.goal_xy is None:
                self._set_status("请先设置终点（点击或输入）")
                return
            self.navigating = True
            self.stop_requested = False
            self.goal_revision += 1
            g = self.goal_xy
        self._set_status(f"开始前往 ({g[0]:.2f}, {g[1]:.2f})")

    def poll(self):
        # 以滑块与输入框的较大者为准，避免只拖一侧时不同步
        try:
            v_num = float(self._speed_num.value)
            v_sld = float(self._speed_slider.value)
            # 若两者接近，用平均值避免浮点抖；否则取最近一次意图（幅值更大的一侧常是用户刚拖的）
            if abs(v_num - v_sld) < 1e-3:
                self.nav_speed = v_num
            else:
                self.nav_speed = v_sld if abs(v_sld - self.nav_speed) >= abs(v_num - self.nav_speed) else v_num
                self._speed_num.value = self.nav_speed
                self._speed_slider.value = self.nav_speed
        except Exception:
            pass
        with self._lock:
            return {
                "goal_xy": self.goal_xy,
                "navigating": self.navigating,
                "goal_revision": self.goal_revision,
                "stop_requested": self.stop_requested,
                "nav_speed": float(self.nav_speed),
                "restart_requested": self.restart_requested,
                "pending_scene": self.pending_scene,
                "reset_requested": self.reset_requested,
            }

    def consume_reset(self):
        with self._lock:
            self.reset_requested = False

    def mark_arrived(self):
        with self._lock:
            self.navigating = False
        g = self.goal_xy
        if g is None:
            self._set_status("已到达")
        else:
            self._set_status(f"已到达 ({g[0]:.2f}, {g[1]:.2f})，可设新终点继续")

    def update_robot(self, xy, yaw=0.0, z=0.20):
        """占位方块更新（网格模式请用 update_robot_from_env）。"""
        if self._robot_handle is None:
            return
        half = yaw * 0.5
        self._robot_handle.position = (float(xy[0]), float(xy[1]), float(z))
        self._robot_handle.wxyz = (math.cos(half), 0.0, 0.0, math.sin(half))

    def update_robot_from_env(self, env):
        """用 Genesis 各连杆位姿驱动 Go2 网格（姿态/腿部与仿真一致）。"""
        try:
            pos = env.base_pos[0].detach().cpu()
            self._robot_xy = (float(pos[0]), float(pos[1]))
        except Exception:
            pass
        if not self._body_handles:
            pos = env.base_pos[0].detach().cpu()
            self.update_robot(
                (float(pos[0]), float(pos[1])),
                yaw=_world_yaw(env),
                z=float(pos[2]),
            )
            return

        try:
            links = env.robot.links
            if self._link_name_to_idx is None:
                self._link_name_to_idx = [lnk.name for lnk in links]
            link_pos = env.robot.get_links_pos()
            link_quat = env.robot.get_links_quat()
            # 形状：[n_env, n_link, 3/4] 或 [n_link, 3/4]
            if link_pos.ndim == 3:
                link_pos = link_pos[0]
                link_quat = link_quat[0]
            link_pos = link_pos.detach().cpu().numpy()
            link_quat = link_quat.detach().cpu().numpy()
        except Exception as exc:
            pos = env.base_pos[0].detach().cpu()
            self.update_robot(
                (float(pos[0]), float(pos[1])),
                yaw=_world_yaw(env),
                z=float(pos[2]),
            )
            if not getattr(self, "_link_warn", False):
                print(f"  [viser] 连杆位姿读取失败，回退刚体方块/基座位姿：{exc}")
                self._link_warn = True
            return

        matched = 0
        with self.server.atomic():
            for i, name in enumerate(self._link_name_to_idx):
                handle = self._body_handles.get(name)
                if handle is None:
                    continue
                matched += 1
                p = link_pos[i]
                q = link_quat[i]
                handle.position = (float(p[0]), float(p[1]), float(p[2]))
                # Genesis quat = (w, x, y, z)，与 viser wxyz 一致
                handle.wxyz = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        if not getattr(self, "_matched_logged", False):
            print(f"  [viser] 网格连杆对齐：{matched}/{len(self._body_handles)} "
                  f"（仿真 links={len(self._link_name_to_idx)}）")
            self._matched_logged = True

    def update_path(self, path):
        if self._path_line is not None:
            try:
                self._path_line.remove()
            except Exception:
                pass
            self._path_line = None
        if not path or len(path) < 2:
            return
        import numpy as np
        pts = np.array([[float(p[0]), float(p[1]), 0.07] for p in path], dtype=np.float32)
        self._path_line = self.server.scene.add_spline_catmull_rom(
            "/planned_path",
            points=pts,
            color=(40, 200, 255),
            line_width=3.0,
            segments=max(16, len(pts) * 2),
        )

    def stop(self):
        try:
            self.server.stop()
        except Exception:
            pass


def _yaw_to_quat(yaw):
    half = yaw * 0.5
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _build_nav_obstacles():
    """侧墙 + 柱障碍（仅仿真场景，规划仍纯 Lidar）。"""
    obstacles = []
    z = NAV_OBSTACLE_H * 0.5
    for wall in NAV_WALL_SPECS:
        obstacles.append({
            "pos": (wall["pos"][0], wall["pos"][1], z),
            "size": wall["size"],
        })
    psize = (NAV_PILLAR_SIZE, NAV_PILLAR_SIZE, NAV_OBSTACLE_H)
    for x, y in NAV_PILLAR_POSITIONS:
        obstacles.append({"pos": (x, y, z), "size": psize})
    return obstacles


def _wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _command_limits(command_cfg):
    return {
        "vx": command_cfg["lin_vel_x_range"],
        "vy": command_cfg["lin_vel_y_range"],
        "w": command_cfg["ang_vel_range"],
    }


def _world_yaw(env):
    """世界系 yaw（弧度）。注意 base_euler 是相对初始朝向，不能直接当世界 yaw 用。"""
    q = env.base_quat[0].detach().cpu()
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _sensor_xy_yaw(env):
    pos = env.base_pos[0].detach().cpu()
    yaw = _world_yaw(env)
    ox, oy = NAV_LIDAR_BODY_OFFSET
    sx = float(pos[0]) + math.cos(yaw) * ox - math.sin(yaw) * oy
    sy = float(pos[1]) + math.sin(yaw) * ox + math.cos(yaw) * oy
    return sx, sy, yaw


def _read_lidar_scan(env):
    """读取完整 Lidar 扫描：命中点 + 每条射线的自由空间端点。"""
    if env.lidar is None:
        return [], []

    sx, sy, yaw = _sensor_xy_yaw(env)
    data = env.lidar.read()
    points = data.points
    distances = data.distances

    if env.num_envs == 1 and points.ndim == 4:
        points = points[0]
        distances = distances[0]

    points_flat = points.reshape(-1, 3)
    dist_flat = distances.reshape(-1)
    n_rays = min(dist_flat.shape[0], len(NAV_LIDAR_AZIMUTHS))

    hits = []
    rays = []
    for i in range(n_rays):
        dist = float(dist_flat[i].item())
        az = NAV_LIDAR_AZIMUTHS[i]
        dx = math.cos(yaw + az)
        dy = math.sin(yaw + az)
        if dist < NAV_LIDAR_RANGE - 1e-4:
            ex = float(points_flat[i, 0].item())
            ey = float(points_flat[i, 1].item())
            hits.append((ex, ey))
            # 自由空间射线在命中点前截断，避免把障碍内部标成可通行
            stop = max(0.05, dist - NAV_RAY_STOP_MARGIN)
            ex, ey = sx + stop * dx, sy + stop * dy
        else:
            ex = sx + NAV_LIDAR_RANGE * dx
            ey = sy + NAV_LIDAR_RANGE * dy
        rays.append(((sx, sy), (ex, ey)))
    return hits, rays


def _read_lidar_hits(env):
    hits, _ = _read_lidar_scan(env)
    return hits


def _bresenham_cells(x0, y0, x1, y1):
    gx0, gy0 = _world_to_grid((x0, y0))
    gx1, gy1 = _world_to_grid((x1, y1))
    dx = abs(gx1 - gx0)
    dy = abs(gy1 - gy0)
    sx = 1 if gx0 < gx1 else -1
    sy = 1 if gy0 < gy1 else -1
    err = dx - dy
    gx, gy = gx0, gy0
    cells = []
    while True:
        cells.append((gx, gy))
        if gx == gx1 and gy == gy1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            gx += sx
        if e2 < dx:
            err += dx
            gy += sy
    return cells


def _make_occupancy_grid():
    width, height = _grid_shape()
    return [[None for _ in range(height)] for _ in range(width)]


def _update_occupancy_grid(grid, hits, rays, robot_xy):
    """累积 Lidar 地图：命中→占据（柱径 + 小膨胀），射线→确认空闲。"""
    width, height = _grid_shape()
    mark_half = NAV_PILLAR_SIZE * 0.5 + NAV_MAP_INFLATION

    for hit in hits:
        hx, hy = hit[0], hit[1]
        gx0, gy0 = _world_to_grid((hx - mark_half, hy - mark_half))
        gx1, gy1 = _world_to_grid((hx + mark_half, hy + mark_half))
        for ix in range(min(gx0, gx1), max(gx0, gx1) + 1):
            for iy in range(min(gy0, gy1), max(gy0, gy1) + 1):
                if _in_grid((ix, iy), width, height):
                    grid[ix][iy] = True

    for (start, end) in rays:
        for gx, gy in _bresenham_cells(start[0], start[1], end[0], end[1]):
            if not _in_grid((gx, gy), width, height):
                continue
            if grid[gx][gy] is not True:
                grid[gx][gy] = False


def _occupancy_to_blocked(grid, robot_xy, goal_xy, robot_clear=NAV_ROBOT_CLEAR):
    """Lidar 规划：仅「确认占据」不可走；未扫描区域视为可走（走近后 Lidar 会更新）。"""
    width, height = _grid_shape()
    blocked = [[cell is True for cell in row] for row in grid]

    for point, radius in ((robot_xy, robot_clear), (goal_xy, 0.20)):
        gx, gy = _world_to_grid(point)
        clear_cells = max(1, int(math.ceil(radius / NAV_GRID_RESOLUTION)))
        for ix in range(gx - clear_cells, gx + clear_cells + 1):
            for iy in range(gy - clear_cells, gy + clear_cells + 1):
                if _in_grid((ix, iy), width, height):
                    blocked[ix][iy] = False
    return blocked


def _reachable_cells(start_xy, blocked):
    width, height = _grid_shape()
    start = _world_to_grid(start_xy)
    if not _in_grid(start, width, height) or blocked[start[0]][start[1]]:
        return []

    queue = [start]
    seen = {start}
    reachable = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        cell = queue.pop(0)
        reachable.append(cell)
        for dx, dy in neighbors:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not _in_grid(nxt, width, height) or blocked[nxt[0]][nxt[1]] or nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return reachable


def _plan_path(start_xy, goal_xy, blocked):
    path = _astar(start_xy, goal_xy, blocked)
    if path is not None:
        return path, True

    reachable = _reachable_cells(start_xy, blocked)
    if not reachable:
        return None, False

    goal = _world_to_grid(goal_xy)
    best = min(
        reachable,
        key=lambda cell: math.hypot(cell[0] - goal[0], cell[1] - goal[1]),
    )
    if best == _world_to_grid(start_xy):
        return None, False
    return _astar(start_xy, _grid_to_world(best), blocked), False


def _grid_shape():
    xmin, xmax, ymin, ymax = NAV_GRID_BOUNDS
    width = int(round((xmax - xmin) / NAV_GRID_RESOLUTION)) + 1
    height = int(round((ymax - ymin) / NAV_GRID_RESOLUTION)) + 1
    return width, height


def _world_to_grid(point):
    xmin, _, ymin, _ = NAV_GRID_BOUNDS
    gx = int(round((point[0] - xmin) / NAV_GRID_RESOLUTION))
    gy = int(round((point[1] - ymin) / NAV_GRID_RESOLUTION))
    return gx, gy


def _grid_to_world(cell):
    xmin, _, ymin, _ = NAV_GRID_BOUNDS
    return (
        xmin + cell[0] * NAV_GRID_RESOLUTION,
        ymin + cell[1] * NAV_GRID_RESOLUTION,
    )


def _in_grid(cell, width, height):
    return 0 <= cell[0] < width and 0 <= cell[1] < height


def _astar(start_xy, goal_xy, occupied):
    width, height = _grid_shape()
    start = _world_to_grid(start_xy)
    # 安全网：终点格映射越界时先钳到地图内再规划
    goal_xy, _, _ = _project_goal_into_world(goal_xy, robot_xy=start_xy)
    goal = _world_to_grid(goal_xy)
    if not _in_grid(start, width, height):
        return None
    if not _in_grid(goal, width, height):
        goal = (
            _clamp(goal[0], 0, width - 1),
            _clamp(goal[1], 0, height - 1),
        )

    neighbors = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    def heuristic(cell):
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    open_heap = [(heuristic(start), 0.0, start)]
    came_from = {}
    cost_so_far = {start: 0.0}

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current == goal:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            cells.reverse()
            return [_grid_to_world(cell) for cell in cells]

        if current_cost > cost_so_far[current]:
            continue

        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not _in_grid(nxt, width, height) or occupied[nxt[0]][nxt[1]]:
                continue
            step_cost = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
            new_cost = current_cost + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(open_heap, (new_cost + heuristic(nxt), new_cost, nxt))

    return None


def _pick_tracking_target(path, robot_xy, blocked):
    """沿路径弧长选取前瞻点（比欧氏距离更稳，避免之字路径上选到侧向点）。"""
    if not path:
        return NAV_GOAL_POINT

    def _free(point):
        gx, gy = _world_to_grid(point)
        width, height = _grid_shape()
        return _in_grid((gx, gy), width, height) and not blocked[gx][gy]

    free_path = [p for p in path if _free(p)]
    if not free_path:
        return path[-1]

    nearest_i = min(
        range(len(free_path)),
        key=lambda i: math.hypot(
            free_path[i][0] - robot_xy[0],
            free_path[i][1] - robot_xy[1],
        ),
    )
    acc = 0.0
    for j in range(nearest_i, len(free_path) - 1):
        acc += math.hypot(
            free_path[j + 1][0] - free_path[j][0],
            free_path[j + 1][1] - free_path[j][1],
        )
        if acc >= NAV_LOOKAHEAD:
            return free_path[j + 1]
    return free_path[-1]


def _apply_hit_repulsion(env, robot_xy, hits, command_cfg, track_target=None):
    """极近距 Lidar 命中才排斥（通道侧墙在 0.7m 内会持续推离，导致原地抖）。"""
    if not hits:
        return
    rep_wx = rep_wy = 0.0
    for hx, hy in hits:
        dx = robot_xy[0] - hx
        dy = robot_xy[1] - hy
        dist = math.hypot(dx, dy)
        if dist >= NAV_HIT_REPULSE_DIST or dist < 0.05:
            continue
        w = (NAV_HIT_REPULSE_DIST - dist) / NAV_HIT_REPULSE_DIST
        rep_wx += w * dx / dist
        rep_wy += w * dy / dist
    if abs(rep_wx) < 1e-6 and abs(rep_wy) < 1e-6:
        return

    yaw = _world_yaw(env)
    body_vx = float(env.commands[0, 0].detach().cpu())
    body_vy = float(env.commands[0, 1].detach().cpu())
    world_vx = math.cos(yaw) * body_vx - math.sin(yaw) * body_vy
    world_vy = math.sin(yaw) * body_vx + math.cos(yaw) * body_vy
    world_vx += NAV_HIT_REPULSE_GAIN * rep_wx
    world_vy += NAV_HIT_REPULSE_GAIN * rep_wy
    body_vx = math.cos(yaw) * world_vx + math.sin(yaw) * world_vy
    body_vy = -math.sin(yaw) * world_vx + math.cos(yaw) * world_vy
    limits = _command_limits(command_cfg)
    env.commands[:, 0] = _clamp(body_vx, *limits["vx"])
    env.commands[:, 1] = _clamp(body_vy, *limits["vy"])
    env._update_observation()


def _nav_replan(env, occupancy_grid, robot_xy):
    scan_hits, scan_rays = _read_lidar_scan(env)
    _update_occupancy_grid(occupancy_grid, scan_hits, scan_rays, robot_xy)
    blocked = _occupancy_to_blocked(occupancy_grid, robot_xy, NAV_GOAL_POINT)
    planned_path, goal_reachable = _plan_path(robot_xy, NAV_GOAL_POINT, blocked)
    return scan_hits, scan_rays, blocked, planned_path, goal_reachable


def _set_command_for_target(env, target, command_cfg, speed):
    """全向速度跟踪：尽量让机体系指令幅值贴近设定 speed（受训练限幅约束）。"""
    pos = env.base_pos[0].detach().cpu()
    yaw = _world_yaw(env)
    dx = target[0] - float(pos[0])
    dy = target[1] - float(pos[1])
    dist = math.hypot(dx, dy)
    limits = _command_limits(command_cfg)

    if dist <= 1e-6:
        env.commands.zero_()
        env._update_observation()
        return dist, env.get_observations()

    desired_yaw = math.atan2(dy, dx)
    yaw_err = _wrap_to_pi(desired_yaw - yaw)

    # 朝向偏差不大时不做降速，避免「滑块=1 但起步明显偏慢」；大转角才减速
    if abs(yaw_err) > math.radians(40.0):
        align = max(0.45, math.cos(yaw_err))
    else:
        align = 1.0

    # 世界系前进方向 → 机体系单位方向
    ux, uy = dx / dist, dy / dist
    bux = math.cos(yaw) * ux + math.sin(yaw) * uy
    buy = -math.sin(yaw) * ux + math.cos(yaw) * uy

    # 在训练速度限幅内，取「沿该机体系方向」的最大可行幅值
    s_lim = float("inf")
    for comp, (lo, hi) in ((bux, limits["vx"]), (buy, limits["vy"])):
        if abs(comp) < 1e-9:
            continue
        if comp > 0.0:
            s_lim = min(s_lim, hi / comp)
        else:
            s_lim = min(s_lim, lo / comp)
    if not math.isfinite(s_lim) or s_lim < 0.0:
        s_lim = 0.0

    cmd_speed = min(float(speed) * align, s_lim)
    body_vx = cmd_speed * bux
    body_vy = cmd_speed * buy
    yaw_rate = 1.1 * yaw_err

    env.commands[:, 0] = _clamp(body_vx, *limits["vx"])
    env.commands[:, 1] = _clamp(body_vy, *limits["vy"])
    env.commands[:, 2] = _clamp(yaw_rate, *limits["w"])
    env._update_observation()
    return dist, env.get_observations()


def _draw_nav_guides(scene, path=None, scan_hits=None, target=None):
    try:
        scene.clear_debug_objects()
        scene.draw_debug_line(
            start=(NAV_START_POINT[0], NAV_START_POINT[1], 0.035),
            end=(NAV_GOAL_POINT[0], NAV_GOAL_POINT[1], 0.035),
            radius=0.004,
            color=(0.8, 0.8, 0.8, 0.7),
        )
        if path and len(path) > 1:
            for start, end in zip(path[:-1], path[1:]):
                scene.draw_debug_line(
                    start=(start[0], start[1], 0.06),
                    end=(end[0], end[1], 0.06),
                    radius=0.01,
                    color=(0.1, 0.7, 1.0, 1.0),
                )
        if scan_hits:
            for hit in scan_hits[:: max(1, len(scan_hits) // 24)]:
                scene.draw_debug_sphere(
                    pos=(hit[0], hit[1], 0.08),
                    radius=0.025,
                    color=(1.0, 0.1, 0.1, 1.0),
                )
        if target is not None:
            scene.draw_debug_line(
                start=(target[0], target[1], 0.03),
                end=(target[0], target[1], 0.25),
                radius=0.01,
                color=(1.0, 0.8, 0.0, 1.0),
            )
        scene.draw_debug_arrow(
            pos=(NAV_GOAL_POINT[0], NAV_GOAL_POINT[1], 0.08),
            vec=(0.0, 0.0, 0.25),
            color=(0.0, 1.0, 0.0, 1.0),
        )
    except Exception:
        # 不同 Genesis 版本的 debug 绘制 API 可能略有差异，绘制失败不影响仿真。
        pass


def load_exp_cfg(log_dir):
    cfgs_pkl = os.path.join(log_dir, "cfgs.pkl")
    if not os.path.isfile(cfgs_pkl):
        raise FileNotFoundError(f"{log_dir} 里没有 cfgs.pkl")

    with open(cfgs_pkl, "rb") as f:
        saved = pickle.load(f)

    env_cfg     = saved["env_cfg"]
    obs_cfg     = saved["obs_cfg"]
    reward_cfg  = saved["reward_cfg"]
    command_cfg = saved["command_cfg"]
    train_cfg   = saved["train_cfg"]
    dr_cfg      = saved.get("dr_cfg", None)

    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, dr_cfg


def pick_ckpt(log_dir, ckpt_num=None):
    ckpts = [f for f in os.listdir(log_dir)
             if f.startswith("model_") and f.endswith(".pt")]
    if not ckpts:
        raise FileNotFoundError(f"{log_dir} 里没有 model_*.pt")
    ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    if ckpt_num is None:
        num = int(ckpts[-1].split("_")[1].split(".")[0])
        print(f"  [ckpt] 自动选最新: {ckpts[-1]}")
        return num
    return ckpt_num


def _reexec_with_nav_scene(scene_name, nav_speed=None):
    """切换场景：带新 --nav_scene 重启当前进程（Genesis 场景无法热替换）。"""
    import sys

    # newtest 入口：python -m newtest.nav.app
    if any("newtest.nav" in (a or "") for a in sys.argv) or (
        len(sys.argv) > 0 and sys.argv[0].endswith("nav/app.py")
    ):
        out = [sys.executable, "-m", "newtest.nav.app"]
    else:
        out = [sys.executable, os.path.abspath(sys.argv[0])]
    i = 1
    argv = sys.argv
    while i < len(argv):
        a = argv[i]
        if a == "--nav_scene" or a == "--nav_speed" or a == "--backend":
            i += 2
            continue
        if a.startswith("--nav_scene=") or a.startswith("--nav_speed=") or a.startswith("--backend="):
            i += 1
            continue
        out.append(a)
        i += 1
    out.extend(["--nav_scene", scene_name])
    if nav_speed is not None:
        out.extend(["--nav_speed", f"{float(nav_speed):.3f}"])
    if "-m" in out and "newtest.nav.app" in out and "--backend" not in out:
        out.extend(["--backend", "multivel"])
    print(f"  [viser] 切换场景 → {scene_name}，保留速度={float(nav_speed):.2f} m/s，重启进程…", flush=True)
    os.execv(sys.executable, out)


def _run_interactive_nav(env, policy, command_cfg, args, gs):
    """Viser 网页交互循环：待机 → 设终点 → 导航 → 到达后继续等待。"""
    ui = NavViserUI(
        port=args.viser_port,
        scene_name=NAV_SCENE_NAME,
        nav_speed=float(args.nav_speed),
    )
    print(f"  [viser] 当前导航设定速度 = {float(args.nav_speed):.2f} m/s "
          f"（来自 --nav_speed / 场景切换保留）", flush=True)
    obs_dict = env.reset()
    env.commands.zero_()
    env._update_observation()
    obs_dict = env.get_observations()

    planned_path = None
    scan_hits = []
    blocked_map = None
    tracking_target = None
    occupancy_grid = _make_occupancy_grid()
    last_revision = -1
    goal_reached_frames = 0
    fall_count = 0
    step = 0

    print("  [viser] 交互循环中（Ctrl+C 退出）...", flush=True)
    try:
        with torch.no_grad():
            # 预热一步，让官方 Lidar 有有效读数
            obs_dict, _, _, _ = env.step(policy(obs_dict))
            step += 1

            while True:
                state = ui.poll()
                if state["restart_requested"] and state["pending_scene"]:
                    scene_key = state["pending_scene"]
                    speed = state["nav_speed"]
                    ui.stop()
                    _reexec_with_nav_scene(scene_key, nav_speed=speed)
                    return None  # execv 成功则不会到这里

                if state.get("reset_requested"):
                    ui.consume_reset()
                    print("  [viser] Reset：回到起点并清空建图", flush=True)
                    planned_path = None
                    scan_hits = []
                    blocked_map = None
                    tracking_target = None
                    occupancy_grid = _make_occupancy_grid()
                    last_revision = -1
                    goal_reached_frames = 0
                    obs_dict = env.reset()
                    env.commands.zero_()
                    env._update_observation()
                    obs_dict = env.get_observations()
                    ui.update_path(None)
                    ui._set_status("重置完成：已回起点，请重新设置终点")
                    ui.server.flush()
                    continue

                robot_xy = (
                    float(env.base_pos[0, 0].detach().cpu()),
                    float(env.base_pos[0, 1].detach().cpu()),
                )
                ui.update_robot_from_env(env)
                nav_speed = float(state["nav_speed"])

                if state["goal_xy"] is not None and state["goal_revision"] != last_revision:
                    projected = _set_nav_goal_xy(state["goal_xy"], robot_xy=robot_xy)
                    env.set_nav_goal(projected)
                    last_revision = state["goal_revision"]
                    occupancy_grid = _make_occupancy_grid()
                    planned_path = None
                    blocked_map = None
                    goal_reached_frames = 0
                    if state["navigating"]:
                        print(f"  [viser] 新终点 ({NAV_GOAL_POINT[0]:.2f}, {NAV_GOAL_POINT[1]:.2f})",
                              flush=True)

                navigating = state["navigating"] and state["goal_xy"] is not None

                # 先规划再发指令，避免「本帧无路径就停车」导致卡死
                if navigating and (planned_path is None or step % NAV_REPLAN_INTERVAL == 0):
                    scan_hits, _, blocked_map, planned_path, goal_reachable = _nav_replan(
                        env, occupancy_grid, robot_xy
                    )
                    ui.update_path(planned_path)
                    if planned_path is None and (step < 3 or step % 50 == 0):
                        print(f"    step {step:5d}  暂无路径，继续扫描 "
                              f"goal=({NAV_GOAL_POINT[0]:.2f},{NAV_GOAL_POINT[1]:.2f})",
                              flush=True)
                    elif step % 100 == 0 and planned_path is not None:
                        tag = "直达" if goal_reachable else "前沿"
                        print(f"    step {step:5d}  path={len(planned_path)} ({tag})  "
                              f"goal=({NAV_GOAL_POINT[0]:.2f},{NAV_GOAL_POINT[1]:.2f})  "
                              f"speed={nav_speed:.2f}",
                              flush=True)

                if navigating:
                    dist_to_goal = math.hypot(
                        NAV_GOAL_POINT[0] - robot_xy[0],
                        NAV_GOAL_POINT[1] - robot_xy[1],
                    )
                    if dist_to_goal < NAV_GOAL_REACH_DIST:
                        env.commands.zero_()
                        env._update_observation()
                        obs_dict = env.get_observations()
                        goal_reached_frames += 1
                        tracking_target = NAV_GOAL_POINT
                        if goal_reached_frames >= 50:
                            ui.mark_arrived()
                            print(f"  [viser] 到达终点 ({NAV_GOAL_POINT[0]:.2f}, "
                                  f"{NAV_GOAL_POINT[1]:.2f})", flush=True)
                            goal_reached_frames = 0
                    elif planned_path is None or blocked_map is None:
                        env.commands.zero_()
                        env._update_observation()
                        obs_dict = env.get_observations()
                        tracking_target = None
                    else:
                        tracking_target = _pick_tracking_target(
                            planned_path, robot_xy, blocked_map
                        )
                        _, obs_dict = _set_command_for_target(
                            env, tracking_target, command_cfg, speed=nav_speed
                        )
                        _apply_hit_repulsion(env, robot_xy, scan_hits, command_cfg)
                        obs_dict = env.get_observations()
                else:
                    env.commands.zero_()
                    env._update_observation()
                    obs_dict = env.get_observations()
                    tracking_target = None

                actions = policy(obs_dict)
                obs_dict, _, dones, infos = env.step(actions)
                step += 1

                _draw_nav_guides(env.scene, planned_path, scan_hits, tracking_target)

                time_out = infos.get("time_outs", torch.zeros(1, device=gs.device))
                is_fall = dones[0].item() and not time_out[0].item()
                if is_fall:
                    fall_count += 1
                    print(f"    摔倒重置（step {step}，累计 {fall_count}）", flush=True)
                    occupancy_grid = _make_occupancy_grid()
                    planned_path = None
                    blocked_map = None
                    obs_dict = env.reset()
                    env.commands.zero_()
                    env._update_observation()
                    obs_dict = env.get_observations()

                # 让出一点时间给 viser 事件（避免占满 GPU/CPU）
                if step % 5 == 0:
                    ui.server.flush()

    except KeyboardInterrupt:
        print("\n  [viser] 用户中断，退出交互循环", flush=True)
    finally:
        try:
            ui.stop()
        except Exception:
            pass

    print(f"  [viser] 结束：共 {step} 步，摔倒 {fall_count} 次")
    return None


def run_eval(exp_name, args, gs):
    interactive = bool(args.nav_demo and getattr(args, "viewer", "native") == "viser")
    print(f"\n{'='*55}")
    if interactive:
        mode = "[Viser交互导航]"
    elif args.nav_demo:
        mode = "[Lidar绕障]"
    elif args.demo:
        mode = "[演示模式]"
    else:
        mode = "[普通模式]"
    print(f"  {exp_name}  {mode}")
    print(f"{'='*55}")

    log_dir = args.log_dir if args.log_dir else f"logs/{exp_name}"
    if not os.path.isdir(log_dir):
        print(f"  [ERROR] 目录不存在: {log_dir}")
        return None

    try:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, dr_cfg = load_exp_cfg(log_dir)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    # eval 时不计算奖励
    reward_cfg["reward_scales"] = {}
    if args.nav_demo:
        nav_goal_id = getattr(args, "nav_goal_id", None) or args.nav_goal
        if interactive:
            scene_name = args.nav_scene
            _apply_nav_scene(scene_name, goal_id=None)
            print(f"  [init] Viser 交互场景={scene_name} ({NAV_SCENES[scene_name]['label']})，"
                  f"障碍将由网页点击/坐标输入指定终点", flush=True)
        else:
            scene_name = "corridor"
            _apply_nav_scene(scene_name, goal_id=nav_goal_id)
            goal_label = _corridor_goal_cfg(nav_goal_id)["label"]
            print(f"  [init] 场景=corridor ({NAV_SCENES['corridor']['label']})，"
                  f"终点={nav_goal_id} ({goal_label})，"
                  f"柱径 {NAV_PILLAR_SIZE:.2f}m", flush=True)

        obstacles = _build_nav_obstacles()
        n_walls = len(NAV_WALL_SPECS)
        n_pillars = len(NAV_PILLAR_POSITIONS)
        print(f"  [init] 障碍 {len(obstacles)} 个（{n_walls} 墙 + {n_pillars} 柱）", flush=True)
        env_cfg["demo_obstacles"] = obstacles
        env_cfg["use_official_lidar"] = True
        env_cfg["freeze_commands"] = True
        start_yaw = math.atan2(
            NAV_GOAL_POINT[1] - NAV_START_POINT[1],
            NAV_GOAL_POINT[0] - NAV_START_POINT[0],
        )
        env_cfg["base_init_pos"] = [
            NAV_START_POINT[0],
            NAV_START_POINT[1],
            env_cfg.get("base_init_pos", [0.0, 0.0, 0.42])[2],
        ]
        env_cfg["base_init_quat"] = _yaw_to_quat(start_yaw)
        env_cfg["lidar_cfg"] = {
            # 与 lidar_teleop.py 一致：pos_offset=(0.3,0,0.1)，2D 水平 360° 扫描
            "fov": (360.0, 0.0),
            "n_points": (NAV_LIDAR_RAYS, 1),
            "max_range": NAV_LIDAR_RANGE,
            "pos_offset": (0.3, 0.0, NAV_LIDAR_SCAN_Z),
            "draw_debug": not interactive,
        }
        # 训练默认 episode=20s，nav 演示需更长，避免约 1000 步超时复位回起点
        if interactive:
            env_cfg["episode_length_s"] = 3600.0  # 交互模式长时间待机
        else:
            env_cfg["episode_length_s"] = args.nav_frames * 0.02 + 5.0

    try:
        ckpt_num = pick_ckpt(log_dir, args.ckpt)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    # ── 相机注入 ──────────────────────────────────────────────────
    original_build = gs.Scene.build
    cam_holder = {}

    def patched_build(self, *a, **kw):
        if args.nav_demo:
            sx, sy = NAV_START_POINT
            if NAV_SCENE_NAME == "fence":
                cam_pos = (sx - 0.05, sy - 1.85, 1.70)
                cam_lookat = (sx - 0.10, sy, 0.28)
                cam_fov = 52
            else:
                cam_pos = (sx - 1.2, sy - 2.0, 2.4)
                cam_lookat = (sx, sy, 0.35)
                cam_fov = 52
        else:
            cam_pos = (1.5, -1.5, 0.8)
            cam_lookat = (0.0, 0.0, 0.3)
            cam_fov = 40
        cam_holder["cam"] = self.add_camera(
            res=(1280, 720),
            pos=cam_pos,
            lookat=cam_lookat,
            fov=cam_fov,
            GUI=False,
        )
        return original_build(self, *a, **kw)

    gs.Scene.build = patched_build

    # ── 创建环境 ──────────────────────────────────────────────────
    print("  [init] 构建场景（scene.build，障碍多时请耐心等待）...", flush=True)
    from go2_env_multivel import Go2EnvMultiVel
    env = Go2EnvMultiVel(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        dr_cfg=None,   # eval 关闭 DR
        show_viewer=False,
    )
    print("  [init] 场景构建完成", flush=True)

    gs.Scene.build = original_build
    cam = cam_holder["cam"]

    follow_axis = NAV_CAMERA_FOLLOW if args.nav_demo else (None, -1.5, 0.8)
    cam.follow_entity(
        env.robot,
        fixed_axis=follow_axis,
        smoothing=0.05,
        fix_orientation=False,
    )

    # ── 加载策略 ──────────────────────────────────────────────────
    print("  [init] 加载策略...", flush=True)
    # newtest 可能把 LeggedGym-Ex 插在 PYTHONPATH 前；multivel 必须用 conda rsl_rl
    import sys as _sys
    _sys.path[:] = [p for p in _sys.path if "LeggedGym-Ex" not in str(p).replace("\\", "/")]
    for _name in list(_sys.modules):
        if _name == "rsl_rl" or _name.startswith("rsl_rl."):
            del _sys.modules[_name]
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    ckpt_path = os.path.join(log_dir, f"model_{ckpt_num}.pt")
    runner.load(ckpt_path, map_location=str(gs.device))
    policy = runner.get_inference_policy(device=gs.device)
    if gs.device.type == "cpu":
        print(f"  [load] model_{ckpt_num}.pt  (CPU 模式)")
    else:
        print(f"  [load] model_{ckpt_num}.pt")

    # ── Viser 交互导航（网页点选/输入终点）────────────────────────
    if interactive:
        return _run_interactive_nav(env, policy, command_cfg, args, gs)

    # ── 录像 ──────────────────────────────────────────────────────
    if args.nav_demo:
        nav_goal_id = getattr(args, "nav_goal_id", None) or args.nav_goal
        suffix = f"nav_demo_corridor_{nav_goal_id}"
    else:
        suffix = "demo" if args.demo else "eval"
    output_path = os.path.abspath(
        os.path.join(log_dir, f"{suffix}_{ckpt_num}.mp4")
    )

    dt = env.dt
    obs_dict = env.reset()

    cam.start_recording()

    if args.nav_demo:
        planned_path = None
        scan_hits = []
        scan_rays = []
        blocked_map = None
        tracking_target = NAV_GOAL_POINT
        occupancy_grid = _make_occupancy_grid()

        nav_goal_id = getattr(args, "nav_goal_id", None) or args.nav_goal
        print("  [nav] scene=%s (%s)  goal_variant=%s (%s)" %
              (NAV_SCENE_NAME, NAV_SCENES[NAV_SCENE_NAME]["label"],
               nav_goal_id, _corridor_goal_cfg(nav_goal_id)["label"]))
        print("  [nav] start=%s  goal=%s  arena=%.1fm  walls=%d  pillars=%d  pillar_size=%.2fm" %
              (NAV_START_POINT, NAV_GOAL_POINT, NAV_ARENA_SIZE,
               len(NAV_WALL_SPECS), len(NAV_PILLAR_POSITIONS), NAV_PILLAR_SIZE))
        print("  [nav] Lidar 建图 + A* 每 %d 帧" % NAV_REPLAN_INTERVAL)
        print("  [nav] lidar: %d rays, range=%.1fm, map_inflate=%.2fm, grid=%.2fm" %
              (NAV_LIDAR_RAYS, NAV_LIDAR_RANGE, NAV_MAP_INFLATION, NAV_GRID_RESOLUTION))
        print("  [nav] 规划规则：Lidar 命中=墙，未扫描=可走（接近时实时更新）")

        total_frames = 0
        fall_count = 0
        goal_reached_frames = 0

        with torch.no_grad():
            # 官方 Lidar 需 sim step 后才有有效扫描
            env.commands.zero_()
            env._update_observation()
            obs_dict = env.get_observations()
            obs_dict, _, _, _ = env.step(policy(obs_dict))
            total_frames += 1
            robot_xy = (
                float(env.base_pos[0, 0].detach().cpu()),
                float(env.base_pos[0, 1].detach().cpu()),
            )
            scan_hits, scan_rays, blocked_map, planned_path, _ = _nav_replan(
                env, occupancy_grid, robot_xy
            )

            for step in range(args.nav_frames):
                robot_xy = (
                    float(env.base_pos[0, 0].detach().cpu()),
                    float(env.base_pos[0, 1].detach().cpu()),
                )
                dist_to_goal = math.hypot(
                    NAV_GOAL_POINT[0] - robot_xy[0],
                    NAV_GOAL_POINT[1] - robot_xy[1],
                )

                if dist_to_goal < NAV_GOAL_REACH_DIST:
                    env.commands.zero_()
                    env._update_observation()
                    obs_dict = env.get_observations()
                    goal_reached_frames += 1
                    tracking_target = NAV_GOAL_POINT
                elif planned_path is None or blocked_map is None:
                    env.commands.zero_()
                    env._update_observation()
                    obs_dict = env.get_observations()
                    tracking_target = None
                else:
                    tracking_target = _pick_tracking_target(
                        planned_path, robot_xy, blocked_map
                    )
                    _, obs_dict = _set_command_for_target(
                        env,
                        tracking_target,
                        command_cfg,
                        speed=args.nav_speed,
                    )
                    _apply_hit_repulsion(env, robot_xy, scan_hits, command_cfg)
                    obs_dict = env.get_observations()

                actions = policy(obs_dict)
                obs_dict, _, dones, infos = env.step(actions)

                robot_xy = (
                    float(env.base_pos[0, 0].detach().cpu()),
                    float(env.base_pos[0, 1].detach().cpu()),
                )
                if planned_path is None or (step + 1) % NAV_REPLAN_INTERVAL == 0:
                    scan_hits, scan_rays, blocked_map, planned_path, goal_reachable = (
                        _nav_replan(env, occupancy_grid, robot_xy)
                    )
                    if planned_path is None:
                        print(f"    step {step+1:4d}  未找到路径，等待扫描")
                    elif step < 2 or (step + 1) % 75 == 0:
                        known_free = sum(
                            1 for row in occupancy_grid for cell in row if cell is False
                        )
                        reach_tag = "直达" if goal_reachable else "前沿"
                        print(f"    step {step+1:4d}  命中 {len(scan_hits)}  已知空闲 {known_free}  "
                              f"路径 {len(planned_path)} 点 ({reach_tag})")

                _draw_nav_guides(env.scene, planned_path, scan_hits, tracking_target)
                cam.render()
                total_frames += 1

                if (step + 1) % 50 == 0:
                    pos = env.base_pos[0].detach().cpu()
                    cmd = env.commands[0].detach().cpu()
                    path_len = 0 if planned_path is None else len(planned_path)
                    tgt = "None" if tracking_target is None else (
                        f"({tracking_target[0]:.2f},{tracking_target[1]:.2f})"
                    )
                    print(f"    step {step+1:4d}  path={path_len:3d}  "
                          f"goal_dist={dist_to_goal:.2f}  pos=({pos[0]:.2f},{pos[1]:.2f})  "
                          f"tgt={tgt}  cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})")

                time_out = infos.get("time_outs", torch.zeros(1, device=gs.device))
                is_fall = dones[0].item() and not time_out[0].item()
                is_timeout = dones[0].item() and time_out[0].item()
                if is_fall or is_timeout:
                    if is_fall:
                        fall_count += 1
                        print(f"    摔倒后重置，重新扫描并规划（step {step+1}）")
                    else:
                        print(f"    episode 超时重置（step {step+1}，"
                              f"原训练 episode=20s，已改为 {env_cfg['episode_length_s']:.0f}s）")
                    planned_path = None
                    scan_hits = []
                    occupancy_grid = _make_occupancy_grid()
                    goal_reached_frames = 0
                    if is_fall:
                        obs_dict = env.reset()

                if goal_reached_frames >= 75:
                    print(f"  [nav] 到达目标点并稳定停留 {goal_reached_frames} 帧")
                    break

        print(f"\n  Lidar 绕障演示完成：{total_frames}帧 ({total_frames*dt:.1f}s)，摔倒{fall_count}次")

    elif args.demo:
        # ── 演示模式：按序列切换速度 ──────────────────────────────
        total_frames = 0
        fall_count   = 0

        for seg_idx, (vx, vy, w, desc) in enumerate(DEMO_SEQUENCE):
            env.commands[:, 0] = vx
            env.commands[:, 1] = vy
            env.commands[:, 2] = w
            print(f"  [{seg_idx+1:2d}/{len(DEMO_SEQUENCE)}] {desc:20s}  "
                  f"vx={vx:+.1f}  vy={vy:+.1f}  ω={w:+.1f}")

            seg_fell = False
            with torch.no_grad():
                for step in range(args.seg_frames):
                    # 保持速度指令（覆盖env内部的重采样）
                    env.commands[:, 0] = vx
                    env.commands[:, 1] = vy
                    env.commands[:, 2] = w

                    actions = policy(obs_dict)
                    obs_dict, _, dones, infos = env.step(actions)
                    cam.render()
                    total_frames += 1

                    time_out = infos.get("time_outs", torch.zeros(1, device=gs.device))
                    is_fall  = dones[0].item() and not time_out[0].item()

                    if is_fall and not seg_fell:
                        print(f"             ⚠ 摔倒（step {step}），重置继续...")
                        fall_count += 1
                        seg_fell = True
                        # 演示模式：摔倒后重置并继续当前段
                        obs_dict = env.reset()
                        env.commands[:, 0] = vx
                        env.commands[:, 1] = vy
                        env.commands[:, 2] = w

        print(f"\n  演示完成：{total_frames}帧 ({total_frames*dt:.1f}s)，摔倒{fall_count}次")

    else:
        # ── 普通模式：随机速度，连续录制 ──────────────────────────
        print(f"  [rec] {args.frames} 帧 ({args.frames*dt:.1f}s)")
        fall_count   = 0
        total_frames = 0

        with torch.no_grad():
            for step in range(args.frames):
                actions = policy(obs_dict)
                obs_dict, _, dones, infos = env.step(actions)
                cam.render()
                total_frames += 1

                if (step + 1) % 100 == 0:
                    pos = env.base_pos[0].cpu()
                    cmd = env.commands[0].cpu()
                    vel = env.base_lin_vel[0].cpu()
                    print(f"    step {step+1:4d}  "
                          f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})  "
                          f"vel_x={vel[0]:+.2f}  "
                          f"pos=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.2f})")

                time_out = infos.get("time_outs", torch.zeros(1, device=gs.device))
                is_fall  = dones[0].item() and not time_out[0].item()
                if is_fall:
                    fall_count += 1
                    obs_dict = env.reset()

        print(f"\n  完成：{total_frames}帧，摔倒{fall_count}次")

    cam.stop_recording(save_to_filename=output_path, fps=50)
    print(f"  ✅ {output_path}")
    return output_path


def main():
    args = get_args()

    import genesis as gs
    backend = gs.cpu if args.cpu else gs.gpu
    gs.init(backend=backend, precision="32", logging_level="warning")
    if backend == gs.gpu and gs.backend == gs.cpu:
        print("  [warn] CUDA 不可用，已回退 CPU；可用 --cpu 显式指定")

    saved = []
    interactive = bool(args.nav_demo and args.viewer == "viser")
    for exp_name in args.exp_name:
        if interactive:
            # Viser：单次长交互会话，终点由网页指定
            path = run_eval(exp_name, args, gs)
            if path:
                saved.append(path)
        elif args.nav_demo:
            goal_ids = [args.nav_goal] if args.nav_goal else [g["id"] for g in NAV_CORRIDOR_GOALS]
            for goal_id in goal_ids:
                run_args = argparse.Namespace(**{**vars(args), "nav_goal_id": goal_id})
                path = run_eval(exp_name, run_args, gs)
                if path:
                    saved.append(path)
        else:
            path = run_eval(exp_name, args, gs)
            if path:
                saved.append(path)

    if saved:
        print(f"\n{'='*55}")
        print("  复制到 Windows：")
        for p in saved:
            print(f"  cp '{p}' /mnt/d/RL/Genesis/")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()