"""导航演示：完整 Lidar 建图 + A* + 近距排斥；行走策略为 go2_wtw。

用法:
  ./newtest/run.sh nav --port 8082
  ./newtest/run.sh nav --cpu --port 8082 --nav_scene corridor --nav_speed 0.65
"""

from __future__ import annotations

import newtest.bootstrap  # noqa: F401

import argparse
import math
import os
import sys
from typing import Dict, List, Optional

import torch

from newtest.nav.load_nav import load_nav_eval_module
from newtest.nav.ui import NavViserUI
from newtest.paths import (
    ASSETS_DIR,
    LEGGED_GYM_ROOT,
    MULTIVEL_LOG_DIR,
    MULTIVEL_LOG_DIR_FALLBACK,
    VENDOR_LEGGED_GYM,
    setup_runtime_paths,
)


def parse_args():
    p = argparse.ArgumentParser(description="导航演示（Lidar 规划 + go2_wtw 行走）")
    p.add_argument("--port", type=int, default=8082)
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument(
        "--backend",
        choices=("wtw", "multivel"),
        default="wtw",
        help="wtw=导航+go2_wtw 行走（默认）；multivel=原 change 完整链路",
    )
    p.add_argument("--nav_scene", type=str, default="corridor", choices=("corridor", "fence", "open8"))
    # Viser 滑条 0.10–1.00；默认 0.65（与参考导航接近）
    p.add_argument("--nav_speed", type=float, default=0.65)
    p.add_argument("--load_run", type=str, default=None, help="go2_wtw run 名")
    p.add_argument("--ckpt", type=int, default=5000, help="checkpoint（wtw 默认 5000；multivel 可改）")
    p.add_argument("--log_dir", type=str, default=None, help="仅 multivel：日志目录")
    p.add_argument("--exp_name", type=str, default="go2-multivel_morestable")
    return p.parse_args()


def _patch_nav_module_for_newtest(nav_mod, *, default_backend: str = "wtw") -> None:
    def _reexec_with_nav_scene(scene_name, nav_speed=None):
        out = [sys.executable, "-m", "newtest.nav.app"]
        i = 1
        argv = sys.argv
        skip_flags = {"--nav_scene", "--nav_speed", "--backend"}
        while i < len(argv):
            a = argv[i]
            if a in skip_flags:
                i += 2
                continue
            if any(a.startswith(f"{f}=") for f in skip_flags):
                i += 1
                continue
            out.append(a)
            i += 1
        out.extend(["--backend", default_backend, "--nav_scene", scene_name])
        if nav_speed is not None:
            out.extend(["--nav_speed", f"{float(nav_speed):.3f}"])
        print(
            f"  [viser] 切换场景 → {scene_name}，保留速度={float(nav_speed):.2f} m/s，重启进程…",
            flush=True,
        )
        os.execv(sys.executable, out)

    _orig_find = nav_mod._find_go2_mjcf

    def _find_go2_mjcf():
        candidates = [
            ASSETS_DIR / "go2" / "go2.xml",
            VENDOR_LEGGED_GYM / "resources" / "robots" / "unitree_robotics" / "go2" / "go2.xml",
            LEGGED_GYM_ROOT / "resources" / "robots" / "unitree_robotics" / "go2" / "go2.xml",
        ]
        for path in candidates:
            if path.is_file():
                return str(path.resolve())
        return _orig_find()

    nav_mod._reexec_with_nav_scene = _reexec_with_nav_scene
    nav_mod._find_go2_mjcf = _find_go2_mjcf


def _resolve_multivel_log_dir(args) -> str:
    if args.log_dir:
        return os.path.abspath(args.log_dir)
    for bundled in (MULTIVEL_LOG_DIR, MULTIVEL_LOG_DIR_FALLBACK):
        if (bundled / "cfgs.pkl").is_file() and any(bundled.glob("model_*.pt")):
            return str(bundled.resolve())
    raise FileNotFoundError(
        "找不到 multivel 日志目录（需含 cfgs.pkl 与 model_*.pt）。\n"
        f"已查找: {MULTIVEL_LOG_DIR} , {MULTIVEL_LOG_DIR_FALLBACK}"
    )


def _prefer_conda_rsl_rl() -> None:
    sys.path[:] = [
        p for p in sys.path if "LeggedGym-Ex" not in str(p).replace("\\", "/")
    ]
    for name in list(sys.modules):
        if name == "rsl_rl" or name.startswith("rsl_rl."):
            del sys.modules[name]


def _command_cfg_from_limits(limits: Dict[str, List[float]]) -> Dict[str, List[float]]:
    return {
        "lin_vel_x_range": limits["vx"],
        "lin_vel_y_range": limits["vy"],
        "ang_vel_range": limits["w"],
    }


def _seed_static_occupancy(nav_mod, grid) -> None:
    """Lidar 不可用时的回退：用已知墙/柱几何填充占据栅格。"""
    width, height = nav_mod._grid_shape()
    inflate = nav_mod.NAV_MAP_INFLATION
    for wall in nav_mod.NAV_WALL_SPECS:
        cx, cy = wall["pos"]
        sx, sy, _ = wall["size"]
        hx, hy = sx * 0.5 + inflate, sy * 0.5 + inflate
        gx0, gy0 = nav_mod._world_to_grid((cx - hx, cy - hy))
        gx1, gy1 = nav_mod._world_to_grid((cx + hx, cy + hy))
        for ix in range(min(gx0, gx1), max(gx0, gx1) + 1):
            for iy in range(min(gy0, gy1), max(gy0, gy1) + 1):
                if nav_mod._in_grid((ix, iy), width, height):
                    grid[ix][iy] = True
    half = nav_mod.NAV_PILLAR_SIZE * 0.5 + inflate
    for px, py in nav_mod.NAV_PILLAR_POSITIONS:
        gx0, gy0 = nav_mod._world_to_grid((px - half, py - half))
        gx1, gy1 = nav_mod._world_to_grid((px + half, py + half))
        for ix in range(min(gx0, gx1), max(gx0, gx1) + 1):
            for iy in range(min(gy0, gy1), max(gy0, gy1) + 1):
                if nav_mod._in_grid((ix, iy), width, height):
                    grid[ix][iy] = True
    for ix in range(width):
        for iy in range(height):
            if grid[ix][iy] is None:
                grid[ix][iy] = False


class _WtwNavBridge:
    """让 nav_mod 的 Lidar / 规划 / 排斥 API 适配 LeggedGym go2_wtw。"""

    def __init__(self, env, lidar):
        self._env = env
        self.lidar = lidar
        self.num_envs = 1
        self.robot = env.simulator._robot

    @property
    def base_pos(self):
        return self._env.simulator.base_pos

    @property
    def base_quat(self):
        # Genesis 连杆/机体系为 wxyz，与 nav_mod._world_yaw 一致
        return self._env.simulator._robot.get_quat()

    @property
    def commands(self):
        return self._env.commands

    def _update_observation(self):
        return None

    def get_observations(self):
        return self._env.get_observations()


def _step_legged(env, actions):
    out = env.step(actions.detach())
    if len(out) >= 5:
        return out[0], out[1], out[2], out[3], out[4]
    obs, rews, dones, infos = out
    return obs, None, rews, dones, infos


def _realign_wtw_pose(env, nav_mod, z0: Optional[float] = None) -> None:
    sx, sy = nav_mod.NAV_START_POINT
    if z0 is None:
        z0 = float(env.simulator.base_pos[0, 2].detach().cpu())
    gx, gy = nav_mod.NAV_GOAL_POINT
    yaw0 = math.atan2(gy - sy, gx - sx)
    half = yaw0 * 0.5
    qwxyz = [math.cos(half), 0.0, 0.0, math.sin(half)]
    import genesis as gs

    robot = env.simulator._robot
    robot.set_quat(
        torch.tensor([qwxyz], dtype=gs.tc_float, device=gs.device),
        zero_velocity=True,
    )
    robot.set_pos(
        torch.tensor([[sx, sy, z0]], dtype=gs.tc_float, device=gs.device),
        zero_velocity=True,
    )
    if hasattr(env.simulator, "_env_origins"):
        env.simulator._env_origins.zero_()
    env.commands.zero_()


def run_wtw_backend(args):
    """导航保留（Lidar 建图 + A* + 近距排斥），行走用 go2_wtw。"""
    from newtest.nav.wtw_env import create_wtw_nav_env

    setup_runtime_paths()
    nav_mod = load_nav_eval_module()
    _patch_nav_module_for_newtest(nav_mod, default_backend="wtw")

    env, policy, meta = create_wtw_nav_env(
        nav_mod,
        scene_name=args.nav_scene,
        cpu=args.cpu,
        load_run=args.load_run,
        checkpoint=args.ckpt if args.ckpt is not None else 5000,
    )
    command_cfg = _command_cfg_from_limits(meta["command_limits"])
    bridge = _WtwNavBridge(env, meta.get("lidar"))
    use_lidar = bridge.lidar is not None

    ui = NavViserUI(
        nav_mod,
        port=args.port,
        scene_name=args.nav_scene,
        nav_speed=float(args.nav_speed),
    )
    print(f"[nav] 后端=wtw（Lidar 规划 + go2_wtw 行走） lidar={'on' if use_lidar else 'static-fallback'}")
    print(
        f"[nav] 场景={args.nav_scene}  速度={float(args.nav_speed):.2f} m/s "
        f"（Viser 可调 0.10–1.00；指令限幅 "
        f"vx={command_cfg['lin_vel_x_range']}）"
    )
    print(f"[nav] Viser: http://localhost:{args.port}")
    print("[nav] 操作：点击地面 / 输入坐标 → 开始导航；网页里可再拉高速度滑条")

    obs = env.get_observations()
    planned_path = None
    blocked_map = None
    scan_hits: List = []
    occupancy_grid = nav_mod._make_occupancy_grid()
    if not use_lidar:
        _seed_static_occupancy(nav_mod, occupancy_grid)
    last_revision = -1
    goal_reached_frames = 0
    step = 0
    z0 = float(env.simulator.base_pos[0, 2].detach().cpu())

    try:
        with torch.no_grad():
            # 预热一步，让官方 Lidar 有有效读数
            env.commands.zero_()
            actions = policy(obs.detach())
            obs, _, _, _, _ = _step_legged(env, actions)
            step += 1

            while True:
                state = ui.poll()
                if state["restart_requested"] and state["pending_scene"]:
                    scene_key = state["pending_scene"]
                    speed = state["nav_speed"]
                    ui.stop()
                    nav_mod._reexec_with_nav_scene(scene_key, nav_speed=speed)
                    return

                if state.get("reset_requested"):
                    if hasattr(ui, "consume_reset"):
                        ui.consume_reset()
                    print("[nav-wtw] Reset：回到起点并清空建图", flush=True)
                    planned_path = None
                    blocked_map = None
                    scan_hits = []
                    occupancy_grid = nav_mod._make_occupancy_grid()
                    if not use_lidar:
                        _seed_static_occupancy(nav_mod, occupancy_grid)
                    last_revision = -1
                    goal_reached_frames = 0
                    obs = env.reset()
                    _realign_wtw_pose(env, nav_mod, z0=z0)
                    obs = env.get_observations()
                    ui.update_path(None)
                    continue

                robot_xy = (
                    float(env.simulator.base_pos[0, 0].detach().cpu()),
                    float(env.simulator.base_pos[0, 1].detach().cpu()),
                )
                try:
                    ui.update_robot_from_env(bridge)
                except Exception:
                    yaw = nav_mod._world_yaw(bridge)
                    ui.update_robot(
                        robot_xy,
                        yaw=yaw,
                        z=float(env.simulator.base_pos[0, 2]),
                    )

                nav_speed = float(state["nav_speed"])
                # 最终钳位由 _set_command_for_target → command_cfg 完成（已放宽到约 ±1.0）

                if state["goal_xy"] is not None and state["goal_revision"] != last_revision:
                    projected = nav_mod._set_nav_goal_xy(state["goal_xy"], robot_xy=robot_xy)
                    last_revision = state["goal_revision"]
                    occupancy_grid = nav_mod._make_occupancy_grid()
                    if not use_lidar:
                        _seed_static_occupancy(nav_mod, occupancy_grid)
                    planned_path = None
                    blocked_map = None
                    goal_reached_frames = 0
                    if state["navigating"]:
                        print(
                            f"[nav-wtw] 新终点 ({nav_mod.NAV_GOAL_POINT[0]:.2f}, "
                            f"{nav_mod.NAV_GOAL_POINT[1]:.2f})",
                            flush=True,
                        )

                navigating = bool(state["navigating"] and state["goal_xy"] is not None)

                if navigating and (planned_path is None or step % nav_mod.NAV_REPLAN_INTERVAL == 0):
                    if use_lidar:
                        scan_hits, _, blocked_map, planned_path, goal_reachable = nav_mod._nav_replan(
                            bridge, occupancy_grid, robot_xy
                        )
                    else:
                        blocked_map = nav_mod._occupancy_to_blocked(
                            occupancy_grid, robot_xy, nav_mod.NAV_GOAL_POINT
                        )
                        planned_path, goal_reachable = nav_mod._plan_path(
                            robot_xy, nav_mod.NAV_GOAL_POINT, blocked_map
                        )
                        scan_hits = []
                    ui.update_path(planned_path)
                    if hasattr(ui, "update_lidar"):
                        ui.update_lidar(scan_hits)
                    if hasattr(ui, "update_occupancy") and step % max(1, nav_mod.NAV_REPLAN_INTERVAL) == 0:
                        ui.update_occupancy(occupancy_grid)
                    if planned_path is None and (step < 3 or step % 50 == 0):
                        print(
                            f"[nav-wtw] step {step}: 暂无路径 "
                            f"robot=({robot_xy[0]:.2f},{robot_xy[1]:.2f}) "
                            f"goal=({nav_mod.NAV_GOAL_POINT[0]:.2f},{nav_mod.NAV_GOAL_POINT[1]:.2f})",
                            flush=True,
                        )
                    elif planned_path is not None and step % 100 == 0:
                        tag = "直达" if goal_reachable else "前沿"
                        print(
                            f"[nav-wtw] step {step}: path={len(planned_path)} ({tag}) "
                            f"speed={nav_speed:.2f}",
                            flush=True,
                        )

                if navigating:
                    dist_to_goal = math.hypot(
                        nav_mod.NAV_GOAL_POINT[0] - robot_xy[0],
                        nav_mod.NAV_GOAL_POINT[1] - robot_xy[1],
                    )
                    if dist_to_goal < nav_mod.NAV_GOAL_REACH_DIST:
                        env.commands.zero_()
                        goal_reached_frames += 1
                        if goal_reached_frames >= 50:
                            ui.mark_arrived()
                            print("[nav-wtw] 到达终点", flush=True)
                            goal_reached_frames = 0
                    elif planned_path is None or blocked_map is None:
                        env.commands.zero_()
                    else:
                        tracking_target = nav_mod._pick_tracking_target(
                            planned_path, robot_xy, blocked_map
                        )
                        nav_mod._set_command_for_target(
                            bridge, tracking_target, command_cfg, speed=nav_speed
                        )
                        if use_lidar:
                            nav_mod._apply_hit_repulsion(
                                bridge, robot_xy, scan_hits, command_cfg
                            )
                        if env.commands.shape[1] > 3:
                            env.commands[:, 3] = 0.0
                else:
                    env.commands.zero_()

                actions = policy(obs.detach())
                obs, _, _, dones, _infos = _step_legged(env, actions)
                step += 1

                if dones is not None and bool(dones[0].item()):
                    infos = _infos if isinstance(_infos, dict) else {}
                    time_out = infos.get("time_outs", None)
                    is_timeout = False
                    if time_out is not None:
                        try:
                            is_timeout = bool(time_out[0].item())
                        except Exception:
                            is_timeout = False
                    if is_timeout:
                        # 超时复位：保留导航地图与终点，仅回位姿
                        if step % 200 == 0:
                            print(f"[nav-wtw] episode 超时复位（step {step}）", flush=True)
                        _realign_wtw_pose(env, nav_mod, z0=z0)
                        obs = env.get_observations()
                    else:
                        print(f"[nav-wtw] 摔倒重置（step {step}），清空建图回起点", flush=True)
                        env.commands.zero_()
                        occupancy_grid = nav_mod._make_occupancy_grid()
                        if not use_lidar:
                            _seed_static_occupancy(nav_mod, occupancy_grid)
                        planned_path = None
                        blocked_map = None
                        _realign_wtw_pose(env, nav_mod, z0=z0)
                        obs = env.get_observations()

                if step % 5 == 0:
                    ui.server.flush()
    except KeyboardInterrupt:
        print("\n[nav-wtw] 退出")
    finally:
        try:
            ui.stop()
        except Exception:
            pass


def run_multivel_backend(args):
    """可选：原 change multivel 完整链路。"""
    setup_runtime_paths()
    nav_mod = load_nav_eval_module()
    _patch_nav_module_for_newtest(nav_mod, default_backend="multivel")
    _prefer_conda_rsl_rl()

    import genesis as gs

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")

    log_dir = _resolve_multivel_log_dir(args)

    class _A:
        pass

    a = _A()
    a.nav_demo = True
    a.viewer = "viser"
    a.viser_port = args.port
    a.nav_scene = args.nav_scene
    a.nav_speed = float(args.nav_speed)
    a.nav_goal = None
    a.nav_goal_id = None
    a.demo = False
    a.ckpt = None if args.ckpt == 5000 else args.ckpt
    a.log_dir = log_dir
    a.cpu = args.cpu
    a.frames = 600
    a.seg_frames = 200
    a.nav_frames = 1500

    print("[nav] 后端=multivel（参考链路）")
    print(f"[nav] 日志目录: {a.log_dir}")
    print(f"[nav] Viser: http://localhost:{args.port}")
    _prefer_conda_rsl_rl()
    nav_mod.run_eval(args.exp_name, a, gs)


def main():
    args = parse_args()
    os.environ.setdefault("SIMULATOR", "genesis")
    if args.backend == "wtw":
        run_wtw_backend(args)
    else:
        run_multivel_backend(args)


if __name__ == "__main__":
    main()
