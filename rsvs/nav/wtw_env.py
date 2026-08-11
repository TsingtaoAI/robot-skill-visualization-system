"""在不修改 LeggedGym-Ex 源码的前提下，注入导航障碍 + 官方 Lidar，并创建 go2_wtw 环境。"""

from __future__ import annotations

import math
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from newtest.common.import_utils import ensure_sys_path
from newtest.paths import (
    LEGGED_GYM_ROOT,
    WTW_DEFAULT_RUN,
    WTW_EXPERIMENT,
    WTW_LOG_RUN_DIR,
    WTW_WEIGHT,
    setup_runtime_paths,
)


def _yaw_to_quat_wxyz(yaw: float) -> List[float]:
    half = yaw * 0.5
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def ensure_wtw_run_logs(*, load_run: str = WTW_DEFAULT_RUN, ckpt: int = 5000) -> None:
    """保证 vendor logs 下有 go2_wtw 权重（仅依赖 newtest）。"""
    run_dir = LEGGED_GYM_ROOT / "logs" / WTW_EXPERIMENT / load_run
    run_dir.mkdir(parents=True, exist_ok=True)
    if ckpt == -1:
        if list(run_dir.glob("model_*.pt")):
            return
        if not WTW_WEIGHT.is_file():
            raise FileNotFoundError(f"缺少 go2_wtw 权重: {WTW_WEIGHT}")
        shutil.copy2(WTW_WEIGHT, run_dir / "model_5000.pt")
        return
    dest = run_dir / f"model_{ckpt}.pt"
    if dest.is_file():
        return
    if WTW_WEIGHT.is_file() and int(ckpt) == 5000:
        shutil.copy2(WTW_WEIGHT, dest)
        return
    raise FileNotFoundError(f"缺少 checkpoint: {dest}")


def create_wtw_nav_env(
    nav_mod,
    *,
    scene_name: str = "corridor",
    cpu: bool = False,
    load_run: Optional[str] = None,
    checkpoint: int = 5000,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """创建 go2_wtw env + inference policy，注入障碍与官方 Lidar。

    Returns:
        env, policy, meta
    """
    setup_runtime_paths()
    ensure_sys_path([LEGGED_GYM_ROOT])
    os.environ.setdefault("SIMULATOR", "genesis")

    # 清掉可能被 nav/multivel 导入的 conda rsl_rl，强制用 LeggedGym-Ex 版
    import sys

    for name in list(sys.modules):
        if name == "rsl_rl" or name.startswith("rsl_rl."):
            del sys.modules[name]

    import genesis as gs

    if not hasattr(gs, "init"):
        raise ImportError(
            f"导入的 genesis 无效: file={getattr(gs, '__file__', None)}。"
            "请在 genesisEx 等已安装 genesis 的 conda 环境中运行。"
        )

    gs.init(backend=gs.cpu if cpu else gs.gpu, logging_level="warning")

    import legged_gym  # noqa: F401
    import legged_gym.envs  # noqa: F401
    from legged_gym.utils import task_registry

    nav_mod._apply_nav_scene(scene_name, goal_id=None)
    obstacles = nav_mod._build_nav_obstacles()

    resolved_load_run = load_run or WTW_DEFAULT_RUN
    resolved_ckpt = 5000 if checkpoint is None else int(checkpoint)
    ensure_wtw_run_logs(load_run=resolved_load_run, ckpt=resolved_ckpt)
    use_cpu = bool(cpu)

    args = type("Args", (), {})()
    args.task = "go2_wtw"
    args.resume = True
    args.load_run = resolved_load_run
    args.checkpoint = resolved_ckpt
    args.ckpt = resolved_ckpt
    args.headless = True
    args.cpu = use_cpu
    args.viewer = "viser"
    args.num_envs = 1
    args.seed = 1
    args.sim_device = "cpu" if use_cpu else "cuda:0"
    args.rl_device = "cpu" if use_cpu else "cuda:0"
    args.max_iterations = None
    args.sync_wandb = False
    args.export_onnx = False
    args.debug = False
    args.use_joystick = False
    args.follow_robot = False
    args.viser_port = 8082
    args.motion_file = None
    args.motion_out_dir = None
    args.num_student = None

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.viewer.rendered_envs_idx = [0]
    env_cfg.env.debug = False
    env_cfg.commands.resampling_time = 1e9
    env_cfg.commands.heading_command = False  # 导航直接写 ang_vel
    env_cfg.commands.zero_cmd_prob = 0.0
    if hasattr(env_cfg.rewards, "behavior_params_range"):
        env_cfg.rewards.behavior_params_range.resampling_time = 1e9
    env_cfg.domain_rand.push_robots = False
    # 导航评估：放宽速度指令限幅（训练默认 vx 仅 ±0.5，Viser 滑条到 1.0 会被白白钳死）
    env_cfg.commands.ranges.lin_vel_x = [-1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [-1.0, 1.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-1.5, 1.5]
    # 导航长时间运行，避免 episode 超时复位
    if hasattr(env_cfg.env, "episode_length_s"):
        env_cfg.env.episode_length_s = 3600.0

    sx, sy = nav_mod.NAV_START_POINT
    gx, gy = nav_mod.NAV_GOAL_POINT
    yaw0 = math.atan2(gy - sy, gx - sx)
    z0 = float(env_cfg.init_state.pos[2])
    env_cfg.init_state.pos = [float(sx), float(sy), z0]

    if hasattr(env_cfg.sim, "max_collision_pairs"):
        env_cfg.sim.max_collision_pairs = max(
            int(getattr(env_cfg.sim, "max_collision_pairs", 64)),
            max(64, len(obstacles) * 4),
        )

    from legged_gym.simulator.genesis_simulator import GenesisSimulator

    _orig_create_envs = GenesisSimulator._create_envs
    lidar_holder: Dict[str, Any] = {}

    def _create_envs_with_nav(self):
        return _patched_create_envs(
            self, _orig_create_envs, obstacles, gs, nav_mod, lidar_holder
        )

    GenesisSimulator._create_envs = _create_envs_with_nav  # type: ignore
    try:
        env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    finally:
        GenesisSimulator._create_envs = _orig_create_envs  # type: ignore

    lidar = lidar_holder.get("lidar")
    env.lidar = lidar  # 供 nav_mod._read_lidar_scan 使用

    try:
        import torch

        if hasattr(env.simulator, "_env_origins"):
            env.simulator._env_origins.zero_()
        qwxyz = _yaw_to_quat_wxyz(yaw0)
        robot = env.simulator._robot
        robot.set_quat(
            torch.tensor([qwxyz], dtype=gs.tc_float, device=gs.device),
            zero_velocity=True,
        )
        robot.set_pos(
            torch.tensor([[sx, sy, z0]], dtype=gs.tc_float, device=gs.device),
            zero_velocity=True,
        )
        if hasattr(env.simulator, "base_init_pos"):
            env.simulator.base_init_pos[:] = torch.tensor(
                [sx, sy, z0],
                dtype=env.simulator.base_init_pos.dtype,
                device=env.simulator.base_init_pos.device,
            )
        if hasattr(env.simulator, "_base_pos"):
            env.simulator._base_pos[0, 0] = sx
            env.simulator._base_pos[0, 1] = sy
            env.simulator._base_pos[0, 2] = z0
        if hasattr(env.simulator, "post_physics_step"):
            try:
                env.simulator.post_physics_step()
            except Exception:
                pass
        pos_now = env.simulator.base_pos[0].detach().cpu()
        print(
            f"[nav-wtw] 世界系对齐: robot→({sx:.2f},{sy:.2f},{z0:.2f}) "
            f"(读回 {float(pos_now[0]):.2f},{float(pos_now[1]):.2f},{float(pos_now[2]):.2f})",
            flush=True,
        )
    except Exception as exc:
        print(f"[nav-wtw] 警告：设置初始位姿失败: {exc}")

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = resolved_load_run
    train_cfg.runner.checkpoint = resolved_ckpt
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root="default"
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    print(
        f"[nav-wtw] 已加载 go2_wtw: run={resolved_load_run} ckpt={resolved_ckpt} "
        f"lidar={'on' if lidar is not None else 'OFF'}",
        flush=True,
    )

    # 稳定默认行为参数（与官方 play 中位附近一致）
    env.commands.zero_()
    if hasattr(env, "gait_period"):
        env.gait_period[:] = 0.45
        env.theta[:, 0] = 0.0
        env.theta[:, 1] = 0.5
        env.theta[:, 2] = 0.5
        env.theta[:, 3] = 0.0
    if hasattr(env, "base_height_target"):
        env.base_height_target[:] = 0.28
    if hasattr(env, "foot_clearance_target"):
        env.foot_clearance_target[:] = 0.08
    if hasattr(env, "pitch_target"):
        env.pitch_target[:] = 0.0

    meta = {
        "scene_name": scene_name,
        "obstacles": obstacles,
        "lidar": lidar,
        "command_limits": {
            "vx": list(env_cfg.commands.ranges.lin_vel_x),
            "vy": list(env_cfg.commands.ranges.lin_vel_y),
            "w": list(env_cfg.commands.ranges.ang_vel_yaw),
        },
        "args": args,
        "env_cfg": env_cfg,
        "log_run_dir": str(WTW_LOG_RUN_DIR),
    }
    return env, policy, meta


def _patched_create_envs(self, orig_create_envs, obstacles, gs, nav_mod, lidar_holder):
    """在 scene.build 前注入障碍 + 官方 Lidar。"""
    scene = self._scene
    orig_build = scene.build

    def build_with_nav(*a, **kw):
        for obs in obstacles:
            pos = obs["pos"]
            size = obs["size"]
            scene.add_entity(
                gs.morphs.Box(
                    pos=(float(pos[0]), float(pos[1]), float(pos[2])),
                    size=(float(size[0]), float(size[1]), float(size[2])),
                    fixed=True,
                )
            )
        print(f"[nav-wtw] 已注入导航障碍 {len(obstacles)} 个", flush=True)

        # 官方 Lidar（与 change 导航一致）
        try:
            pattern = gs.sensors.SphericalPattern(
                fov=(360.0, 0.0),
                n_points=(int(nav_mod.NAV_LIDAR_RAYS), 1),
            )
            lidar = scene.add_sensor(
                gs.sensors.Lidar(
                    pattern=pattern,
                    entity_idx=self._robot.idx,
                    pos_offset=(0.3, 0.0, float(nav_mod.NAV_LIDAR_SCAN_Z)),
                    euler_offset=(0.0, 0.0, 0.0),
                    max_range=float(nav_mod.NAV_LIDAR_RANGE),
                    return_world_frame=True,
                    draw_debug=False,
                )
            )
            lidar_holder["lidar"] = lidar
            print(
                f"[nav-wtw] 已挂载官方 Lidar: rays={nav_mod.NAV_LIDAR_RAYS} "
                f"range={nav_mod.NAV_LIDAR_RANGE:.1f}m",
                flush=True,
            )
        except Exception as exc:
            lidar_holder["lidar"] = None
            print(f"[nav-wtw] Lidar 挂载失败（将回退静态图）: {exc}", flush=True)

        return orig_build(*a, **kw)

    scene.build = build_with_nav
    try:
        return orig_create_envs(self)
    finally:
        scene.build = orig_build
