"""官方对齐：LeggedGym-Ex go2_wtw Viser play。

等价于:
  SIMULATOR=genesis python -m legged_gym.scripts.play \\
      --task go2_wtw --resume --viewer viser

用法:
  ./newtest/run.sh play --port 8083
  ./newtest/run.sh play --cpu --port 8083
"""

from __future__ import annotations

import newtest.bootstrap  # noqa: F401

import argparse
import os
import shutil
import sys
from types import SimpleNamespace

from newtest.paths import (
    LEGGED_GYM_ROOT,
    WTW_DEFAULT_RUN,
    WTW_EXPERIMENT,
    WTW_WEIGHT,
    setup_runtime_paths,
)


def parse_args():
    p = argparse.ArgumentParser(description="go2_wtw 官方对齐行走演示（Viser）")
    p.add_argument("--port", type=int, default=8083, help="Viser 端口（对应 --viser_port）")
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--task", type=str, default="go2_wtw")
    p.add_argument("--load_run", type=str, default=WTW_DEFAULT_RUN)
    p.add_argument("--ckpt", type=int, default=5000, help="checkpoint 编号；-1=目录内最新")
    p.add_argument("--follow_robot", action="store_true", default=False)
    p.add_argument("--use_joystick", action="store_true", default=False)
    p.add_argument("--joystick_type", type=str, default="xbox")
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--export_onnx", action="store_true", default=False)
    return p.parse_args()


def ensure_wtw_run_logs(*, load_run: str, ckpt: int) -> None:
    """保证 LEGGED_GYM_ROOT/logs/go2_wtw/<run>/ 下有可用权重（仅依赖 newtest）。"""
    run_dir = LEGGED_GYM_ROOT / "logs" / WTW_EXPERIMENT / load_run
    run_dir.mkdir(parents=True, exist_ok=True)

    if ckpt == -1:
        existing = sorted(run_dir.glob("model_*.pt"))
        if existing:
            return
        # 无任何 model → 用内置 weights
        if not WTW_WEIGHT.is_file():
            raise FileNotFoundError(
                f"缺少 go2_wtw 权重: {WTW_WEIGHT}\n"
                f"且日志目录为空: {run_dir}"
            )
        dest = run_dir / "model_5000.pt"
        shutil.copy2(WTW_WEIGHT, dest)
        print(f"[play] 已从 weights 注入 {dest}")
        return

    dest = run_dir / f"model_{ckpt}.pt"
    if dest.is_file():
        return
    if WTW_WEIGHT.is_file() and ckpt == 5000:
        shutil.copy2(WTW_WEIGHT, dest)
        print(f"[play] 已从 weights 注入 {dest}")
        return
    # 尝试任意内置 model_5000 作为回退提示
    raise FileNotFoundError(
        f"缺少 checkpoint: {dest}\n"
        f"请将 model_{ckpt}.pt 放入该目录，或使用 --ckpt 5000（对应 weights/go2_wtw/model_5000.pt）"
    )


def _build_play_args(cli) -> SimpleNamespace:
    """构造与 legged_gym.utils.helpers.get_args() 兼容的参数对象。"""
    return SimpleNamespace(
        task=cli.task,
        headless=True,  # viser 模式
        cpu=bool(cli.cpu),
        num_envs=1,
        max_iterations=None,
        resume=True,
        sync_wandb=False,
        export_onnx=bool(cli.export_onnx),
        debug=bool(cli.debug),
        load_run=cli.load_run,
        ckpt=int(cli.ckpt),
        use_joystick=bool(cli.use_joystick),
        joystick_type=cli.joystick_type,
        follow_robot=bool(cli.follow_robot),
        viewer="viser",
        viser_port=int(cli.port),
        motion_file=None,
        motion_out_dir=None,
        num_student=None,
        seed=1,
        sim_device="cpu" if cli.cpu else "cuda:0",
        rl_device="cpu" if cli.cpu else "cuda:0",
    )


def main():
    cli = parse_args()
    os.environ["SIMULATOR"] = "genesis"
    setup_runtime_paths()

    # play 必须用 newtest 内置 rsl_rl（与 go2_wtw 权重匹配）
    # setup_runtime_paths 已把 LEGGED_GYM_ROOT（newtest 根）插到最前
    for name in list(sys.modules):
        if name == "rsl_rl" or name.startswith("rsl_rl."):
            del sys.modules[name]

    ensure_wtw_run_logs(load_run=cli.load_run, ckpt=cli.ckpt)
    args = _build_play_args(cli)

    print("[play] 对齐官方: SIMULATOR=genesis python -m legged_gym.scripts.play "
          f"--task {args.task} --resume --viewer viser")
    print(f"[play] load_run={args.load_run}  ckpt={args.ckpt}  port={args.viser_port}")
    print(f"[play] LEGGED_GYM_ROOT={LEGGED_GYM_ROOT}")
    print(f"[play] Viser: http://localhost:{args.viser_port}")

    # 在官方 play 使用 create_viser_viewer 之前挂载自研控制台
    import legged_gym.utils.viser_viewer as vv_mod
    import legged_gym.scripts.play as play_mod
    from newtest.play.controls import attach_play_console

    # Viser 遥控必须用角速度通道：在 make_env 之前关闭 heading_command，
    # 否则环境会按朝向误差持续改写 yaw → 不停转圈
    _orig_override = play_mod.override_configs

    def _override_for_teleop(env_cfg, args, task_type):
        _orig_override(env_cfg, args, task_type)
        if getattr(args, "viewer", None) == "viser":
            env_cfg.commands.heading_command = False
            env_cfg.commands.zero_cmd_prob = 0.0
            print("[play] override: heading_command=False（Viser 遥控）", flush=True)

    play_mod.override_configs = _override_for_teleop

    _orig_create = vv_mod.create_viser_viewer

    def _create_enhanced(env, port: int = 8080, robot_index: int = 0):
        # 再次确保（防 override 未生效）
        try:
            env.cfg.commands.heading_command = False
            env.commands[:, :3] = 0.0
        except Exception:
            pass
        viewer = _orig_create(env, port=port, robot_index=robot_index)
        return attach_play_console(viewer, env=env)

    vv_mod.create_viser_viewer = _create_enhanced
    play_mod.create_viser_viewer = _create_enhanced

    from legged_gym.scripts.play import play as official_play

    official_play(args)


if __name__ == "__main__":
    main()
