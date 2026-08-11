"""
Chain policies in one Genesis simulation:

  handstand -> hold -> scripted recover -> hold quad
  -> Genesis legstand_cycle (rise/hold/recover) -> hold quad
  -> backflip -> hold quad -> in-place left turn 90° (multivel)
  -> settle -> ramp to 1 m/s -> forward 2s -> stop
  -> spring jump x2 -> backflip_double -> done

Example:
  python examples/locomotion/go2_eval_combo_gym_policy.py --headless
  python examples/locomotion/go2_eval_combo_gym_policy.py --headless --video go2_combo_hs_cycle_flip.mp4
"""

from __future__ import annotations

import argparse
import enum
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CONTROL_DT = 0.005
SIM_SUBSTEPS = 2
DECIMATION = 4  # 200Hz sim -> 50Hz policy for Genesis skills
POLICY_DT = CONTROL_DT * DECIMATION
NUM_ACTIONS = 12
CLIP_ACTIONS = 100.0
CLIP_OBS = 100.0
ACTION_SCALE = 0.25

OBS_STAND = 48
OBS_CYCLE = 46
OBS_BACKFLIP = 60
OBS_TURN = 45  # go2_env_multivel: ang(3)+grav(3)+cmd(3)+dof(12)+vel(12)+act(12)
OBS_JUMP_SINGLE = 47  # go2_spring_jump single-frame obs
JUMP_FRAME_STACK = 10
OBS_JUMP = JUMP_FRAME_STACK * OBS_JUMP_SINGLE  # 470

# Isaac Gym handstand joint order
JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

# Genesis cycle / backflip joint order
GENESIS_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

# Isaac order (FL,FR,RL,RR)
DEFAULT_QUAD_DOF = torch.tensor(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    dtype=torch.float32,
)
HANDSTAND_DESIRED = torch.tensor(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 2.25, -1.75, 0.0, 2.25, -1.75],
    dtype=torch.float32,
)
# Genesis order (FR,FL,RR,RL) — matches go2_legstand_cycle / go2_backflip
CYCLE_QUAD_DOF = torch.tensor(
    [-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5],
    dtype=torch.float32,
)
BACKFLIP_QUAD_DOF = torch.tensor(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=torch.float32,
)
# Multivel default (FR,FL,RR,RL) — hips at 0 like go2_env_multivel
MULTIVEL_QUAD_DOF = torch.tensor(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=torch.float32,
)
# Spring jump default (FL,FR,RL,RR) — matches go2_eval_spring_jump_gym_policy
SPRING_JUMP_DOF = torch.tensor(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
    dtype=torch.float32,
)
# Handstand -> forward prone -> low quad -> stand (scripted, smooth)
PRONE_FORWARD_DOF = torch.tensor(
    [0.0, 0.35, -1.25, 0.0, 0.35, -1.25, 0.05, 1.35, -1.65, -0.05, 1.35, -1.65],
    dtype=torch.float32,
)
QUAD_LOW_DOF = torch.tensor(
    [0.08, 0.95, -1.55, -0.08, 0.95, -1.55, 0.12, 1.05, -1.55, -0.12, 1.05, -1.55],
    dtype=torch.float32,
)

STAND_KP = 40.0
STAND_KD = 1.6
STAND_TAU = 33.5
CYCLE_KP = 40.0
CYCLE_KD = 1.0
CYCLE_TAU = 33.5
CYCLE_ACTION_SCALE = 0.25
CYCLE_RISE_END_S = 8.0
CYCLE_HOLD_END_S = 14.0
CYCLE_EPISODE_S = 30.0
BACKFLIP_KP = 70.0
BACKFLIP_KD = 3.0
BACKFLIP_TAU = 33.5
BACKFLIP_ACTION_SCALE = 0.5
BACKFLIP_EPISODE_S = 2.0
BACKFLIP_EXTRA_S = 0.5
BACKFLIP_DOUBLE_EPISODE_S = 3.0  # go2_backflip -e double
TURN_KP = 20.0
TURN_KD = 0.5
TURN_TAU = 33.5
TURN_ACTION_SCALE = 0.25
TURN_ANG_VEL = 0.8  # 原地左转指令（rad/s）
TURN_YAW_DEG = 90.0  # 目标偏航角变化
TURN_DURATION_S = 4.0  # 超时兜底（按偏航角提前结束）
FORWARD_VX = 1.0  # 前进速度指令（m/s），对齐 multivel 范围
FORWARD_DURATION_S = 2.0
FORWARD_RAMP_S = 0.8  # 0→vx 斜坡，避免阶跃指令栽跟头
POST_TURN_SETTLE_S = 0.6  # 转向后零指令站稳再起步
STOP_DURATION_S = 1.0  # 前进后零指令停下
JUMP_KP = 20.0
JUMP_KD = 0.5
JUMP_TAU = 25.0
JUMP_ACTION_SCALE = 0.25
JUMP_DISTANCE = 1.0
JUMP_FRAME = 55  # policy step when commands[2] flips to 1
JUMP_STARTUP_STEPS = 100  # ~0.5s zero-action warm-up like spring_jump eval
JUMP_DURATION_S = 4.0  # timeout; normally ends earlier once landed
BRIDGE_S = 1.5  # settle between jumps / jump→backflip (1~2s)

INIT_HEIGHT = 0.42


class Phase(enum.Enum):
    HANDSTAND = "handstand"
    HOLD_HANDSTAND = "hold_handstand"
    RECOVER_QUAD = "recover_quad"
    HOLD_QUAD = "hold_quad"
    LEGSTAND_CYCLE = "legstand_cycle"
    HOLD_QUAD_2 = "hold_quad_2"
    BACKFLIP = "backflip"
    HOLD_QUAD_3 = "hold_quad_3"
    TURN_INPLACE = "turn_inplace"
    SETTLE_AFTER_TURN = "settle_after_turn"
    FORWARD = "forward"
    STOP = "stop"
    SPRING_JUMP = "spring_jump"
    HOLD_BRIDGE = "hold_bridge"
    BACKFLIP_FINALE = "backflip_finale"
    DONE = "done"


# Only anchor after pose is stable; never during rise/tilt (causes flip torque).
ANCHOR_HOLD_PHASES = frozenset({Phase.HOLD_HANDSTAND})


@dataclass
class PhaseRuntime:
    phase: Phase = Phase.HANDSTAND
    phase_step: int = 0
    stable_steps: int = 0
    policy_step: int = 0
    stand_warmup_done: bool = False
    recover_warned: bool = False
    recover_initial_pitch: float = 0.0
    recover_initial_quat: Optional[torch.Tensor] = None
    recover_initial_pos: Optional[torch.Tensor] = None
    anchor_pos: Optional[torch.Tensor] = None
    obs_action: torch.Tensor = field(default_factory=lambda: torch.zeros(1, NUM_ACTIONS))
    filtered_target_q: torch.Tensor = field(default_factory=lambda: torch.zeros(1, NUM_ACTIONS))
    target_q: torch.Tensor = field(default_factory=lambda: torch.zeros(1, NUM_ACTIONS))
    stand_hist: deque = field(default_factory=lambda: deque(maxlen=1))
    # Genesis skill buffers (cycle / backflip), FR-FL-RR-RL order
    genesis_action: torch.Tensor = field(default_factory=lambda: torch.zeros(1, NUM_ACTIONS))
    genesis_last_action: torch.Tensor = field(default_factory=lambda: torch.zeros(1, NUM_ACTIONS))
    cycle_stand_command: float = 0.0
    turn_yaw0: Optional[float] = None
    jump_hist: deque = field(default_factory=lambda: deque(maxlen=JUMP_FRAME_STACK))
    jump_policy_step: int = 0
    jump_count: int = 0  # completed spring jumps in the finale sequence

    def reset_stand_policy(self, device):
        self.stand_warmup_done = False
        self.obs_action = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32, device=device)
        self.filtered_target_q = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32, device=device)
        self.target_q = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32, device=device)
        self.stand_hist = deque(
            [torch.zeros((1, OBS_STAND), dtype=torch.float32, device=device)],
            maxlen=1,
        )

    def reset_genesis_policy(self, device):
        self.policy_step = 0
        self.genesis_action = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32, device=device)
        self.genesis_last_action = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32, device=device)
        self.cycle_stand_command = 0.0
        self.turn_yaw0 = None

    def reset_jump_policy(self, device):
        self.jump_policy_step = 0
        self.obs_action = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32, device=device)
        self.target_q = torch.zeros((1, NUM_ACTIONS), dtype=torch.float32, device=device)
        self.jump_hist = deque(
            [torch.zeros((1, OBS_JUMP_SINGLE), dtype=torch.float32, device=device) for _ in range(JUMP_FRAME_STACK)],
            maxlen=JUMP_FRAME_STACK,
        )

    def reset_phase_counters(self):
        self.phase_step = 0
        self.stable_steps = 0
        self.recover_warned = False


def resolve_path(path: str, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else repo_root / candidate


def make_stand_actor():
    return nn.Sequential(
        nn.Linear(OBS_STAND, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, NUM_ACTIONS),
    )


def make_cycle_actor():
    return nn.Sequential(
        nn.Linear(OBS_CYCLE, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, NUM_ACTIONS),
    )


def make_turn_actor():
    return nn.Sequential(
        nn.Linear(OBS_TURN, 512),
        nn.ELU(),
        nn.Linear(512, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, NUM_ACTIONS),
    )


def load_checkpoint_actor(path: Path, device, input_dim: int):
    checkpoint = torch.load(path, map_location="cpu")
    if input_dim == OBS_STAND:
        factory = make_stand_actor
    elif input_dim == OBS_TURN:
        factory = make_turn_actor
    else:
        factory = make_cycle_actor
    # Isaac Gym export style
    if "model_state_dict" in checkpoint:
        actor_state = {
            key[len("actor.") :]: value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("actor.")
        }
        if not actor_state:
            raise KeyError(f"{path} 中没有 actor.* 参数。")
        actor = factory()
        actor.load_state_dict(actor_state)
    # rsl-rl-lib>=5 Genesis cycle / multivel style
    elif "actor_state_dict" in checkpoint:
        raw = checkpoint["actor_state_dict"]
        mlp_state = {k[len("mlp.") :]: v for k, v in raw.items() if k.startswith("mlp.")}
        actor = factory()
        actor.load_state_dict(mlp_state)
    else:
        raise KeyError(f"{path} 无法识别的 checkpoint 格式: {list(checkpoint.keys())}")
    actor.to(device)
    actor.eval()
    return actor


def load_policy(policy_path: Path, checkpoint_path: Optional[Path], device, input_dim: int):
    if policy_path.exists():
        # TorchScript (handstand export / backflip) or raw checkpoint
        try:
            policy = torch.jit.load(str(policy_path), map_location="cpu").to(device)
            policy.eval()
            return policy
        except Exception:
            return load_checkpoint_actor(policy_path, device, input_dim)
    if checkpoint_path is not None and checkpoint_path.exists():
        return load_checkpoint_actor(checkpoint_path, device, input_dim)
    raise FileNotFoundError(f"找不到 policy: {policy_path}")


def compute_torques(target_q, dof_pos, dof_vel, default_dof_pos, kp, kd, tau_limit):
    torques = kp * (target_q + default_dof_pos - dof_pos) - kd * dof_vel
    return torch.clip(torques, -tau_limit, tau_limit)


def hold_pd_torques(dof_pos, dof_vel, target_dof, kp, kd, tau_limit):
    return torch.clip(kp * (target_dof - dof_pos) - kd * dof_vel, -tau_limit, tau_limit)


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def lerp_dof(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    return a * (1.0 - t) + b * t


def recovery_handstand_to_quad_joints(t: float, handstand_joints, prone_joints, low_joints, quad_joints) -> torch.Tensor:
    if t < 0.38:
        return lerp_dof(handstand_joints, prone_joints, smoothstep(t / 0.38))
    if t < 0.72:
        return lerp_dof(prone_joints, low_joints, smoothstep((t - 0.38) / 0.34))
    return lerp_dof(low_joints, quad_joints, smoothstep((t - 0.72) / 0.28))


def set_phase_anchor(rt: PhaseRuntime, robot):
    rt.anchor_pos = robot.get_pos().clone()


def should_apply_anchor(
    rt: PhaseRuntime,
    phase: Phase,
    cfg,
    projected_gravity: Optional[torch.Tensor] = None,
    stand_prog: Optional[torch.Tensor] = None,
    base_ang_vel: Optional[torch.Tensor] = None,
) -> bool:
    if not cfg.anchor_enabled or rt.anchor_pos is None or phase not in ANCHOR_HOLD_PHASES:
        return False
    if projected_gravity is None or stand_prog is None or base_ang_vel is None:
        return True
    upright = float(-projected_gravity[0, 2]) > cfg.anchor_upright_pg_thresh
    calm = float(torch.norm(base_ang_vel[0])) < cfg.anchor_max_ang_vel
    stable = float(stand_prog[0, 0]) > cfg.anchor_min_stand_prog
    return upright and calm and stable


def apply_anchor_drift_control(robot, scene, rt: PhaseRuntime, cfg, device, scale: float = 1.0):
    if rt.anchor_pos is None or scale <= 0.0:
        return
    pos = robot.get_pos()
    vel = robot.get_vel()
    force = torch.zeros((1, 3), dtype=torch.float32, device=device)
    kp = cfg.anchor_pos_kp * scale
    kd = cfg.anchor_vel_kd * scale
    force[0, 0] = kp * (rt.anchor_pos[0, 0] - pos[0, 0]) - kd * vel[0, 0]
    force[0, 1] = kp * (rt.anchor_pos[0, 1] - pos[0, 1]) - kd * vel[0, 1]
    ang_vel = robot.get_ang()
    scene.sim.rigid_solver.apply_links_external_force(
        force=force,
        links_idx=(robot.base_link_idx,),
        envs_idx=(0,),
        ref="link_com",
        local=False,
    )
    yaw_torque = torch.zeros((1, 3), dtype=torch.float32, device=device)
    yaw_torque[0, 2] = -cfg.anchor_yaw_kd * scale * ang_vel[0, 2]
    scene.sim.rigid_solver.apply_links_external_torque(
        torque=yaw_torque,
        links_idx=(robot.base_link_idx,),
        envs_idx=(0,),
        ref="link_com",
        local=False,
    )


def apply_body_level_torque(robot, scene, device, cfg, fade: float = 1.0):
    """Virtual balance controller for scripted recoveries from inverted stands."""
    if fade <= 0.0:
        return
    base_quat = robot.get_quat()
    body_up = transform_by_quat(
        torch.tensor([[0.0, 0.0, 1.0]], dtype=gs.tc_float, device=device),
        base_quat,
    )
    world_up = torch.tensor([[0.0, 0.0, 1.0]], dtype=gs.tc_float, device=device)
    err = torch.cross(body_up, world_up, dim=-1)
    ang_vel = robot.get_ang().to(gs.tc_float)
    torque = (cfg.recover_att_kp * err - cfg.recover_att_kd * ang_vel) * fade
    scene.sim.rigid_solver.apply_links_external_torque(
        torque=torque,
        links_idx=(robot.base_link_idx,),
        envs_idx=(0,),
        ref="link_com",
        local=False,
    )


def run_scripted_recovery(
    rt: PhaseRuntime,
    dof_pos,
    dof_vel,
    cfg,
    joint_target,
    kp,
    kd,
    tau_limit,
    policy_torques=None,
    policy_keep_end: float = 0.30,
    kp_boost: Optional[torch.Tensor] = None,
):
    t = min(1.0, rt.phase_step / max(1, cfg.recover_descend_steps))
    eff_kp = kp * kp_boost if kp_boost is not None else kp
    pd_torques = hold_pd_torques(dof_pos, dof_vel, joint_target, eff_kp, kd * cfg.recover_kd_scale, tau_limit)
    policy_keep = max(0.0, 1.0 - t / max(policy_keep_end, 1e-3))
    if policy_torques is not None and policy_keep > 0.0:
        return policy_keep * policy_torques + (1.0 - policy_keep) * pd_torques
    return pd_torques


def build_stand_obs(robot, motor_dofs, default_dof_pos, last_action, command, device):
    base_quat = robot.get_quat()
    inv_base_quat = inv_quat(base_quat)
    base_ang_vel = transform_by_quat(robot.get_ang(), inv_base_quat)
    projected_gravity = transform_by_quat(
        torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=device),
        inv_base_quat,
    )
    dof_pos = robot.get_dofs_position(motor_dofs)
    dof_vel = robot.get_dofs_velocity(motor_dofs)

    obs = torch.zeros((1, OBS_STAND), dtype=torch.float32, device=device)
    obs[:, 3:6] = base_ang_vel * 0.25
    obs[:, 6:9] = projected_gravity
    obs[:, 9] = command[0] * 2.0
    obs[:, 10] = command[1] * 2.0
    obs[:, 11] = command[2] * 0.25
    obs[:, 12:24] = (dof_pos - default_dof_pos) * 1.0
    obs[:, 24:36] = dof_vel * 0.05
    obs[:, 36:48] = last_action
    return (
        torch.clip(obs, -CLIP_OBS, CLIP_OBS),
        dof_pos,
        dof_vel,
        projected_gravity,
        base_ang_vel,
    )


def build_cycle_obs(robot, motor_dofs, default_dof_pos, last_action, stand_command, command, device):
    """46-dim obs matching go2_legstand_cycle_env (FR,FL,RR,RL)."""
    base_quat = robot.get_quat()
    inv_base_quat = inv_quat(base_quat)
    base_ang_vel = transform_by_quat(robot.get_ang(), inv_base_quat)
    projected_gravity = transform_by_quat(
        torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=device),
        inv_base_quat,
    )
    dof_pos = robot.get_dofs_position(motor_dofs)
    dof_vel = robot.get_dofs_velocity(motor_dofs)

    obs = torch.zeros((1, OBS_CYCLE), dtype=torch.float32, device=device)
    obs[:, 0:1] = stand_command
    obs[:, 1:4] = base_ang_vel * 0.25
    obs[:, 4:7] = projected_gravity
    obs[:, 7] = command[0] * 2.0
    obs[:, 8] = command[1] * 2.0
    obs[:, 9] = command[2] * 0.25
    obs[:, 10:22] = (dof_pos - default_dof_pos) * 1.0
    obs[:, 22:34] = dof_vel * 0.05
    obs[:, 34:46] = last_action
    return (
        torch.clip(obs, -CLIP_OBS, CLIP_OBS),
        dof_pos,
        dof_vel,
        projected_gravity,
        base_ang_vel,
    )


def build_backflip_obs(robot, motor_dofs, default_dof_pos, actions, last_actions, policy_step, max_steps, device):
    """60-dim obs matching go2_backflip._build_backflip_obs (FR,FL,RR,RL)."""
    base_quat = robot.get_quat()
    inv_base_quat = inv_quat(base_quat)
    base_ang_vel = transform_by_quat(robot.get_ang(), inv_base_quat)
    projected_gravity = transform_by_quat(
        torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=device),
        inv_base_quat,
    )
    dof_pos = robot.get_dofs_position(motor_dofs)
    dof_vel = robot.get_dofs_velocity(motor_dofs)

    # Clamp to training horizon so BACKFLIP_EXTRA_S 不把 phase 推到 OOD（对齐 go2_env episode 上限）
    phase = math.pi * float(min(policy_step, max_steps)) / max(1, max_steps)
    obs = torch.zeros((1, OBS_BACKFLIP), dtype=torch.float32, device=device)
    obs[:, 0:3] = base_ang_vel * 0.25
    obs[:, 3:6] = projected_gravity
    obs[:, 6:18] = (dof_pos - default_dof_pos) * 1.0
    obs[:, 18:30] = dof_vel * 0.05
    obs[:, 30:42] = actions
    obs[:, 42:54] = last_actions
    obs[:, 54] = math.sin(phase)
    obs[:, 55] = math.cos(phase)
    obs[:, 56] = math.sin(phase / 2.0)
    obs[:, 57] = math.cos(phase / 2.0)
    obs[:, 58] = math.sin(phase / 4.0)
    obs[:, 59] = math.cos(phase / 4.0)
    return (
        torch.clip(obs, -CLIP_OBS, CLIP_OBS),
        dof_pos,
        dof_vel,
        projected_gravity,
        base_ang_vel,
    )


def build_turn_obs(robot, motor_dofs, default_dof_pos, last_action, command, device):
    """45-dim obs matching go2_env_multivel (FR,FL,RR,RL)."""
    base_quat = robot.get_quat()
    inv_base_quat = inv_quat(base_quat)
    base_ang_vel = transform_by_quat(robot.get_ang(), inv_base_quat)
    projected_gravity = transform_by_quat(
        torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=device),
        inv_base_quat,
    )
    dof_pos = robot.get_dofs_position(motor_dofs)
    dof_vel = robot.get_dofs_velocity(motor_dofs)

    obs = torch.zeros((1, OBS_TURN), dtype=torch.float32, device=device)
    obs[:, 0:3] = base_ang_vel * 0.25
    obs[:, 3:6] = projected_gravity
    obs[:, 6] = command[0] * 2.0
    obs[:, 7] = command[1] * 2.0
    obs[:, 8] = command[2] * 0.25
    obs[:, 9:21] = (dof_pos - default_dof_pos) * 1.0
    obs[:, 21:33] = dof_vel * 0.05
    obs[:, 33:45] = last_action
    return (
        torch.clip(obs, -CLIP_OBS, CLIP_OBS),
        dof_pos,
        dof_vel,
        projected_gravity,
        base_ang_vel,
    )


def wrap_euler(euler: torch.Tensor) -> torch.Tensor:
    return torch.where(euler > math.pi, euler - 2.0 * math.pi, euler)


def build_jump_obs(robot, motor_dofs, default_dof_pos, last_action, command, device, quat_to_xyz):
    """47-dim single-frame obs matching go2_eval_spring_jump_gym_policy (FL,FR,RL,RR)."""
    base_quat = robot.get_quat()
    base_ang_vel = transform_by_quat(robot.get_ang(), inv_quat(base_quat))
    base_euler = wrap_euler(quat_to_xyz(base_quat, rpy=True, degrees=False))
    dof_pos = robot.get_dofs_position(motor_dofs)
    dof_vel = robot.get_dofs_velocity(motor_dofs)

    obs = torch.zeros((1, OBS_JUMP_SINGLE), dtype=torch.float32, device=device)
    obs[:, 2:5] = command.unsqueeze(0)
    obs[:, 5:8] = base_ang_vel * 0.25
    obs[:, 8:11] = base_euler * 1.0
    obs[:, 11:23] = (dof_pos - default_dof_pos) * 1.0
    obs[:, 23:35] = dof_vel * 0.05
    obs[:, 35:47] = last_action
    return torch.clip(obs, -CLIP_OBS, CLIP_OBS), dof_pos, dof_vel


def prefill_jump_history(rt: PhaseRuntime, robot, motor_dofs, spring_default, device, quat_to_xyz, jump_distance):
    command = torch.tensor([jump_distance, 0.0, 0.0], dtype=torch.float32, device=device)
    rt.jump_hist.clear()
    for _ in range(JUMP_FRAME_STACK):
        obs, _, _ = build_jump_obs(
            robot, motor_dofs, spring_default.unsqueeze(0), rt.obs_action, command, device, quat_to_xyz
        )
        rt.jump_hist.append(obs.clone())


def is_policy_step(global_step: int) -> bool:
    return global_step % DECIMATION == 0


def is_handstand_stable(projected_gravity, stand_prog, base_ang_vel, cfg, phase_step: int) -> bool:
    if phase_step < cfg.hs_startup + cfg.hs_ramp:
        return False
    pg_ok = float(-projected_gravity[0, 0]) > cfg.stable_pg_thresh
    prog_ok = float(stand_prog[0, 0]) > cfg.stand_prog_thresh
    calm = float(torch.norm(base_ang_vel[0])) < cfg.stand_ang_vel_thresh
    return pg_ok and prog_ok and calm


def is_upright_quad(projected_gravity: torch.Tensor, dof_pos, target_dof, base_ang_vel, cfg) -> bool:
    pg = projected_gravity[0]
    upright = pg[2] < -cfg.quad_pg_z_thresh and abs(pg[0]) < cfg.quad_pg_xy_thresh and abs(pg[1]) < cfg.quad_pg_xy_thresh
    joint_ok = torch.norm(dof_pos[0] - target_dof) < cfg.quad_joint_thresh
    calm = torch.norm(base_ang_vel[0]) < cfg.quad_ang_vel_thresh
    return bool(upright and joint_ok and calm)


def is_transition_ready(projected_gravity: torch.Tensor, dof_pos, target_dof, base_ang_vel, cfg) -> bool:
    """Loose upright check for brief skill handoffs (not a long pose lock)."""
    pg = projected_gravity[0]
    upright = (
        pg[2] < -cfg.trans_pg_z_thresh
        and abs(pg[0]) < cfg.trans_pg_xy_thresh
        and abs(pg[1]) < cfg.trans_pg_xy_thresh
    )
    joint_ok = torch.norm(dof_pos[0] - target_dof) < cfg.trans_joint_thresh
    calm = torch.norm(base_ang_vel[0]) < cfg.trans_ang_vel_thresh
    return bool(upright and joint_ok and calm)


def is_cycle_landed(robot, projected_gravity: torch.Tensor, base_ang_vel, cfg) -> bool:
    """Roughly back on four feet after cycle recover — no joint matching (avoids 30s timeout freeze)."""
    pg = projected_gravity[0]
    level = float(pg[2]) < -cfg.cycle_land_pg_z
    height = float(robot.get_pos()[0, 2])
    height_ok = cfg.cycle_land_h_min < height < cfg.cycle_land_h_max
    calm = float(torch.norm(base_ang_vel[0])) < cfg.cycle_land_ang_vel
    return bool(level and height_ok and calm)


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(quat: torch.Tensor) -> float:
    """Extract yaw (rad) from wxyz quaternion."""
    q = quat[0]
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def warm_start_genesis_action(rt: PhaseRuntime, robot, motor_dofs, default_dof_pos, action_scale: float):
    """Seed last/current action from current pose so the next skill does not jerk from zeros."""
    dof_pos = robot.get_dofs_position(motor_dofs)
    action = torch.clip((dof_pos - default_dof_pos) / max(action_scale, 1.0e-6), -CLIP_ACTIONS, CLIP_ACTIONS)
    rt.genesis_action = action.clone()
    rt.genesis_last_action = action.clone()


def stand_progress_handstand(projected_gravity, gate_start, gate_end):
    return torch.clamp(
        (-projected_gravity[:, 0] - gate_start) / max(gate_end - gate_start, 1.0e-6),
        0.0,
        1.0,
    ).unsqueeze(1)


def run_stand_phase(
    rt: PhaseRuntime,
    policy,
    robot,
    motor_dofs,
    default_dof_pos,
    desired_delta,
    command,
    device,
    kp,
    kd,
    tau_limit,
    global_step: int,
    cfg,
):
    obs, dof_pos, dof_vel, projected_gravity, base_ang_vel = build_stand_obs(
        robot, motor_dofs, default_dof_pos, rt.obs_action, command, device
    )
    stand_prog = stand_progress_handstand(projected_gravity, cfg.hs_gate_start, cfg.hs_gate_end)

    if is_policy_step(global_step):
        rt.stand_hist.append(obs)
        policy_input = torch.cat(list(rt.stand_hist), dim=1)
        policy_action = torch.clip(policy(policy_input), -CLIP_ACTIONS, CLIP_ACTIONS)
        rt.target_q = policy_action * ACTION_SCALE
        rise_alpha = min(max(cfg.rise_action_filter, 0.0), 0.95)
        stand_alpha = min(max(cfg.stand_action_filter, 0.0), 0.95)
        alpha = rise_alpha * (1.0 - stand_prog) + stand_alpha * stand_prog
        rt.filtered_target_q = alpha * rt.filtered_target_q + (1.0 - alpha) * rt.target_q

    startup = cfg.hs_startup
    ramp_steps = cfg.hs_ramp
    hold_blend = cfg.hold_pose_blend
    in_hold = rt.phase == Phase.HOLD_HANDSTAND
    if in_hold:
        hold_blend = cfg.hold_stand_pose_blend

    if in_hold or rt.stand_warmup_done:
        ramp = 1.0
    elif rt.phase_step < startup:
        ramp = 0.0
    else:
        ramp = min(1.0, (rt.phase_step - startup + 1) / max(1, ramp_steps))
        if ramp >= 1.0:
            rt.stand_warmup_done = True

    if ramp <= 0.0:
        blended_target_q = torch.zeros_like(rt.target_q)
    else:
        hold = min(max(hold_blend, 0.0), 0.65) * ramp * stand_prog
        if in_hold:
            stabilized = (1.0 - hold) * rt.filtered_target_q * cfg.hold_policy_scale + hold * desired_delta
        else:
            stabilized = (1.0 - hold) * rt.filtered_target_q + hold * desired_delta
        blended_target_q = stabilized * ramp

    rt.obs_action = torch.clip(blended_target_q / max(ACTION_SCALE, 1.0e-6), -CLIP_ACTIONS, CLIP_ACTIONS)
    torques = compute_torques(blended_target_q, dof_pos, dof_vel, default_dof_pos, kp, kd, tau_limit)
    return torques, projected_gravity, stand_prog, base_ang_vel


def run_hold_pose(rt, robot, motor_dofs, target_dof, default_dof_pos, device, kp, kd, tau_limit, command):
    obs, dof_pos, dof_vel, projected_gravity, base_ang_vel = build_stand_obs(
        robot, motor_dofs, default_dof_pos, rt.obs_action, command, device
    )
    torques = hold_pd_torques(dof_pos, dof_vel, target_dof, kp, kd, tau_limit)
    hold_delta = target_dof.unsqueeze(0) - default_dof_pos
    rt.obs_action = torch.clip(hold_delta / max(ACTION_SCALE, 1.0e-6), -CLIP_ACTIONS, CLIP_ACTIONS)
    return torques, projected_gravity, base_ang_vel, dof_pos


def run_legstand_cycle_phase(
    rt: PhaseRuntime,
    policy,
    robot,
    motor_dofs,
    default_dof_pos,
    command,
    device,
    kp,
    kd,
    tau_limit,
    global_step: int,
    hold_end_s: float,
    use_position: bool = False,
):
    """Genesis legstand_cycle @ 50Hz (FR,FL,RR,RL).

    Matches go2_legstand_cycle_env: 1-step action latency; optional position control.
    """
    t = rt.policy_step * POLICY_DT
    stand_cmd = 1.0 if t >= hold_end_s else 0.0
    rt.cycle_stand_command = stand_cmd
    stand_command = torch.tensor([[stand_cmd]], dtype=torch.float32, device=device)

    obs, dof_pos, dof_vel, projected_gravity, base_ang_vel = build_cycle_obs(
        robot,
        motor_dofs,
        default_dof_pos,
        rt.genesis_action,
        stand_command,
        command,
        device,
    )

    exec_action = rt.genesis_last_action
    if is_policy_step(global_step):
        new_action = torch.clip(policy(obs), -CLIP_ACTIONS, CLIP_ACTIONS)
        rt.genesis_action = new_action
        rt.genesis_last_action = new_action.clone()
        rt.policy_step += 1

    target_q = exec_action * CYCLE_ACTION_SCALE
    if use_position:
        return target_q + default_dof_pos, projected_gravity, base_ang_vel, dof_pos, stand_cmd, True
    torques = compute_torques(target_q, dof_pos, dof_vel, default_dof_pos, kp, kd, tau_limit)
    return torques, projected_gravity, base_ang_vel, dof_pos, stand_cmd, False


def run_backflip_phase(
    rt: PhaseRuntime,
    policy,
    robot,
    motor_dofs,
    default_dof_pos,
    device,
    kp,
    kd,
    tau_limit,
    global_step: int,
    max_policy_steps: int,
    use_position: bool = False,
):
    """Genesis backflip @ 50Hz with 60-dim phase obs (FR,FL,RR,RL).

    Matches go2_env / go2_backflip step semantics:
    - obs uses actions/last_actions that are equal after each policy step
    - motors execute previous action (1-step latency)
    - when ``use_position`` is True, returns joint position targets
    """
    obs, dof_pos, dof_vel, projected_gravity, base_ang_vel = build_backflip_obs(
        robot,
        motor_dofs,
        default_dof_pos,
        rt.genesis_action,
        rt.genesis_last_action,
        rt.policy_step,
        max_policy_steps,
        device,
    )

    # Capture exec target BEFORE updating buffers (latency)
    exec_action = rt.genesis_last_action
    if is_policy_step(global_step):
        new_action = torch.clip(policy(obs), -CLIP_ACTIONS, CLIP_ACTIONS)
        rt.genesis_action = new_action
        # After this step, obs buffers match go2_env (actions == last_actions)
        rt.genesis_last_action = new_action.clone()
        rt.policy_step += 1

    target_q = exec_action * BACKFLIP_ACTION_SCALE
    if use_position:
        return target_q + default_dof_pos, projected_gravity, base_ang_vel, dof_pos, True
    torques = compute_torques(target_q, dof_pos, dof_vel, default_dof_pos, kp, kd, tau_limit)
    return torques, projected_gravity, base_ang_vel, dof_pos, False


def run_turn_phase(
    rt: PhaseRuntime,
    policy,
    robot,
    motor_dofs,
    default_dof_pos,
    command,
    device,
    kp,
    kd,
    tau_limit,
    global_step: int,
    use_position: bool = False,
):
    """Multivel locomotion @ 50Hz with 45-dim obs (FR,FL,RR,RL).

    When ``use_position`` is True, returns a joint position target matching
    training (``control_dofs_position`` + 1-step action latency). Otherwise
    returns torques from an equivalent PD law.
    """
    obs, dof_pos, dof_vel, projected_gravity, base_ang_vel = build_turn_obs(
        robot,
        motor_dofs,
        default_dof_pos,
        rt.genesis_action,
        command,
        device,
    )

    if is_policy_step(global_step):
        action = torch.clip(policy(obs), -CLIP_ACTIONS, CLIP_ACTIONS)
        rt.genesis_last_action = rt.genesis_action.clone()
        rt.genesis_action = action
        rt.policy_step += 1

    # Match go2_env_multivel: execute previous action (simulate_action_latency).
    exec_action = rt.genesis_last_action
    target_q = exec_action * TURN_ACTION_SCALE
    if use_position:
        target_pos = target_q + default_dof_pos
        return target_pos, projected_gravity, base_ang_vel, dof_pos, True
    torques = compute_torques(target_q, dof_pos, dof_vel, default_dof_pos, kp, kd, tau_limit)
    return torques, projected_gravity, base_ang_vel, dof_pos, False


def run_spring_jump_phase(
    rt: PhaseRuntime,
    policy,
    robot,
    motor_dofs,
    default_dof_pos,
    device,
    kp,
    kd,
    tau_limit,
    global_step: int,
    jump_distance: float,
    jump_frame: int,
    startup_steps: int,
    quat_to_xyz,
):
    """Isaac spring_jump @ 50Hz with 47×10 stacked obs (FL,FR,RL,RR), torque PD."""
    if is_policy_step(global_step):
        rt.jump_policy_step += 1
    jump_flag = 1.0 if rt.jump_policy_step >= jump_frame else 0.0
    command = torch.tensor([jump_distance, 0.0, jump_flag], dtype=torch.float32, device=device)

    obs, dof_pos, dof_vel = build_jump_obs(
        robot,
        motor_dofs,
        default_dof_pos,
        rt.obs_action,
        command,
        device,
        quat_to_xyz,
    )

    if is_policy_step(global_step):
        rt.jump_hist.append(obs)
        policy_input = torch.cat(list(rt.jump_hist), dim=1)
        policy_action = torch.clip(policy(policy_input), -CLIP_ACTIONS, CLIP_ACTIONS)
        rt.target_q = policy_action * JUMP_ACTION_SCALE

    if rt.phase_step < startup_steps:
        blended_target_q = torch.zeros_like(rt.target_q)
    else:
        blended_target_q = rt.target_q

    rt.obs_action = torch.clip(blended_target_q / max(JUMP_ACTION_SCALE, 1.0e-6), -CLIP_ACTIONS, CLIP_ACTIONS)
    torques = compute_torques(blended_target_q, dof_pos, dof_vel, default_dof_pos, kp, kd, tau_limit)
    return torques, dof_pos, dof_vel


def advance_phase(rt: PhaseRuntime, new_phase: Phase, device, reason: str):
    print(f"[combo] {rt.phase.value} -> {new_phase.value} ({reason})")
    rt.phase = new_phase
    rt.reset_phase_counters()
    if new_phase in (Phase.LEGSTAND_CYCLE, Phase.BACKFLIP, Phase.BACKFLIP_FINALE, Phase.TURN_INPLACE):
        rt.reset_genesis_policy(device)
    if new_phase == Phase.HANDSTAND:
        rt.reset_stand_policy(device)
    if new_phase == Phase.SPRING_JUMP:
        rt.reset_jump_policy(device)


def advance_phase_with_robot(
    rt: PhaseRuntime,
    new_phase: Phase,
    device,
    reason: str,
    robot,
    warm_start_dofs=None,
    warm_start_default=None,
    warm_start_scale: Optional[float] = None,
):
    advance_phase(rt, new_phase, device, reason)
    if new_phase in ANCHOR_HOLD_PHASES:
        set_phase_anchor(rt, robot)
    if (
        new_phase in (Phase.LEGSTAND_CYCLE, Phase.BACKFLIP, Phase.BACKFLIP_FINALE, Phase.TURN_INPLACE)
        and warm_start_dofs is not None
        and warm_start_default is not None
        and warm_start_scale is not None
    ):
        warm_start_genesis_action(rt, robot, warm_start_dofs, warm_start_default, warm_start_scale)


def main():
    global gs, inv_quat, transform_by_quat

    parser = argparse.ArgumentParser(
        description="Genesis combo: … → spring_jump x2 → backflip_double."
    )
    parser.add_argument(
        "--handstand_model",
        type=str,
        default="My_unitree_go2_gym/logs/go2_handstand/exported/policies/policy_1.pt",
    )
    parser.add_argument(
        "--legstand_cycle_ckpt",
        type=str,
        default="logs/go2-legstand-cycle/model_4000.pt",
        help="rsl-rl cycle checkpoint (relative to this script or repo root).",
    )
    parser.add_argument(
        "--backflip_model",
        type=str,
        default="backflip/single.pt",
        help="TorchScript single backflip (mid-combo after cycle).",
    )
    parser.add_argument(
        "--backflip_double_model",
        type=str,
        default="backflip/double.pt",
        help="TorchScript double backflip after spring jumps (go2_backflip -e double).",
    )
    parser.add_argument(
        "--multivel_ckpt",
        type=str,
        default="weights/go2_wtw/model_5000.pt",
        help="rsl-rl multivel checkpoint for in-place turn (45-dim obs).",
    )
    parser.add_argument(
        "--turn_ang_vel",
        type=float,
        default=TURN_ANG_VEL,
        help="Yaw-rate command for in-place left turn (rad/s).",
    )
    parser.add_argument(
        "--turn_yaw_deg",
        type=float,
        default=TURN_YAW_DEG,
        help="Stop in-place turn after this many degrees of yaw change (default 90).",
    )
    parser.add_argument(
        "--turn_s",
        type=float,
        default=TURN_DURATION_S,
        help="Timeout for in-place turn if yaw target is not reached.",
    )
    parser.add_argument(
        "--forward_vx",
        type=float,
        default=FORWARD_VX,
        help="Forward lin_vel_x command after turn (m/s).",
    )
    parser.add_argument(
        "--forward_s",
        type=float,
        default=FORWARD_DURATION_S,
        help="Seconds of constant-speed forward after ramp.",
    )
    parser.add_argument(
        "--forward_ramp_s",
        type=float,
        default=FORWARD_RAMP_S,
        help="Seconds to ramp vx from 0 to forward_vx (avoids faceplant on step command).",
    )
    parser.add_argument(
        "--post_turn_settle_s",
        type=float,
        default=POST_TURN_SETTLE_S,
        help="Zero-command settle after turn before starting forward.",
    )
    parser.add_argument(
        "--stop_s",
        type=float,
        default=STOP_DURATION_S,
        help="Seconds of zero-command stop after forward.",
    )
    parser.add_argument(
        "--spring_jump_model",
        type=str,
        default="My_unitree_go2_gym/logs/go2_spring_jump/exported/policies/policy_1.pt",
        help="TorchScript spring_jump policy (same as go2_eval_spring_jump_gym_policy).",
    )
    parser.add_argument("--jump_distance", type=float, default=JUMP_DISTANCE)
    parser.add_argument("--jump_frame", type=int, default=JUMP_FRAME)
    parser.add_argument("--jump_startup_steps", type=int, default=JUMP_STARTUP_STEPS)
    parser.add_argument(
        "--spring_jump_s",
        type=float,
        default=JUMP_DURATION_S,
        help="Max seconds per spring jump (exits earlier once landed).",
    )
    parser.add_argument(
        "--bridge_s",
        type=float,
        default=BRIDGE_S,
        help="Settle between spring jumps and before/between finale backflips (seconds).",
    )
    parser.add_argument("--video", type=str, default="go2_combo_hs_cycle_flip.mp4")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("-c", "--cpu", action="store_true")
    parser.add_argument("--sim_substeps", type=int, default=SIM_SUBSTEPS)
    parser.add_argument("--model_format", choices=("mjcf", "urdf"), default="mjcf")
    parser.add_argument(
        "--mjcf_model",
        type=str,
        default="My_unitree_go2_gym/resources/robots/go2/go2/go2.xml",
    )
    parser.add_argument("--max_steps", type=int, default=None)
    # phase timing
    parser.add_argument("--hold_handstand_s", type=float, default=2.0)
    parser.add_argument(
        "--hold_quad_s",
        type=float,
        default=0.35,
        help="Hard cap for brief quad handoff between skills (not a long freeze).",
    )
    parser.add_argument("--recover_quad_s", type=float, default=8.0)
    parser.add_argument(
        "--stand_stable_s",
        type=float,
        default=1.2,
        help="Seconds stable before leaving handstand skill (not used for quad handoffs).",
    )
    parser.add_argument(
        "--transition_stable_s",
        type=float,
        default=0.18,
        help="Brief settle before handing off to the next skill.",
    )
    parser.add_argument("--handstand_max_s", type=float, default=12.0, help="Max seconds in handstand before forcing hold.")
    parser.add_argument("--cycle_hold_end_s", type=float, default=CYCLE_HOLD_END_S)
    parser.add_argument(
        "--cycle_recover_max_s",
        type=float,
        default=3.0,
        help="After cycle recover starts, force handoff within this many seconds (do not wait full episode).",
    )
    parser.add_argument("--cycle_episode_s", type=float, default=CYCLE_EPISODE_S)
    parser.add_argument("--cycle_land_pg_z", type=float, default=0.55, help="Max projected_gravity_z to count as landed.")
    parser.add_argument("--cycle_land_h_min", type=float, default=0.18)
    parser.add_argument("--cycle_land_h_max", type=float, default=0.55)
    parser.add_argument("--cycle_land_ang_vel", type=float, default=2.5)
    parser.add_argument("--backflip_episode_s", type=float, default=BACKFLIP_EPISODE_S)
    parser.add_argument("--backflip_extra_s", type=float, default=BACKFLIP_EXTRA_S)
    parser.add_argument(
        "--backflip_double_episode_s",
        type=float,
        default=BACKFLIP_DOUBLE_EPISODE_S,
        help="Episode length for double backflip policy (training default 3s).",
    )
    parser.add_argument("--stable_pg_thresh", type=float, default=0.82)
    parser.add_argument("--stand_prog_thresh", type=float, default=0.92)
    parser.add_argument("--stand_ang_vel_thresh", type=float, default=0.8)
    # stand deploy knobs (handstand)
    parser.add_argument("--hs_startup", type=int, default=300)
    parser.add_argument("--hs_ramp", type=int, default=150)
    parser.add_argument("--hs_gate_start", type=float, default=0.45)
    parser.add_argument("--hs_gate_end", type=float, default=0.75)
    parser.add_argument("--rise_action_filter", type=float, default=0.05)
    parser.add_argument("--stand_action_filter", type=float, default=0.30)
    parser.add_argument("--hold_pose_blend", type=float, default=0.08)
    parser.add_argument("--hold_stand_pose_blend", type=float, default=0.38, help="Stronger pose lock during stand holds.")
    parser.add_argument("--hold_policy_scale", type=float, default=0.12, help="Scale policy output during hold to reduce pacing.")
    parser.add_argument("--anchor_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor_pos_kp", type=float, default=80.0, help="XY spring during stable hold only.")
    parser.add_argument("--anchor_vel_kd", type=float, default=14.0, help="XY velocity damping during stable hold.")
    parser.add_argument("--anchor_yaw_kd", type=float, default=8.0, help="Yaw rate damping during stable hold.")
    parser.add_argument("--anchor_upright_pg_thresh", type=float, default=0.75, help="Min -pg_z before anchor engages.")
    parser.add_argument("--anchor_min_stand_prog", type=float, default=0.85, help="Min stand progress before anchor engages.")
    parser.add_argument("--anchor_max_ang_vel", type=float, default=1.2, help="Max body rate before anchor engages.")
    parser.add_argument("--recover_descend_steps", type=int, default=900, help="Scripted recovery duration (~4.5s).")
    parser.add_argument("--recover_kd_scale", type=float, default=1.35, help="KD multiplier for scripted recovery PD.")
    parser.add_argument("--recover_kd_boost", type=float, default=1.5)
    parser.add_argument("--recover_att_kp", type=float, default=18.0, help="Body attitude Kp for physical inverted->level recovery.")
    parser.add_argument("--recover_att_kd", type=float, default=6.0, help="Body attitude Kd for physical inverted->level recovery.")
    parser.add_argument("--recover_rear_kp_boost", type=float, default=2.5, help="Rear-leg KP boost during handstand->quad.")
    parser.add_argument("--kd_scale", type=float, default=1.2)
    # recovery thresholds
    parser.add_argument("--quad_pg_z_thresh", type=float, default=0.88)
    parser.add_argument("--quad_pg_xy_thresh", type=float, default=0.28)
    parser.add_argument("--quad_joint_thresh", type=float, default=0.45)
    parser.add_argument("--quad_ang_vel_thresh", type=float, default=0.9)
    # looser thresholds for brief skill-to-skill handoffs
    parser.add_argument("--trans_pg_z_thresh", type=float, default=0.72)
    parser.add_argument("--trans_pg_xy_thresh", type=float, default=0.40)
    parser.add_argument("--trans_joint_thresh", type=float, default=0.75)
    parser.add_argument("--trans_ang_vel_thresh", type=float, default=1.6)
    args = parser.parse_args()

    args.stable_steps_req = max(1, int(args.stand_stable_s / CONTROL_DT))
    args.transition_steps_req = max(1, int(args.transition_stable_s / CONTROL_DT))
    args.hold_handstand_steps = int(args.hold_handstand_s / CONTROL_DT)
    args.hold_quad_steps = int(args.hold_quad_s / CONTROL_DT)
    args.recover_quad_max = int(args.recover_quad_s / CONTROL_DT)
    args.handstand_max_steps = max(
        int(args.handstand_max_s / CONTROL_DT),
        args.hs_startup + args.hs_ramp + args.stable_steps_req,
    )
    args.cycle_max_steps = int(args.cycle_episode_s / CONTROL_DT)
    args.cycle_recover_max_steps = int(args.cycle_recover_max_s / CONTROL_DT)
    args.backflip_max_policy_steps = max(1, int(args.backflip_episode_s / POLICY_DT))
    args.backflip_max_steps = int((args.backflip_episode_s + args.backflip_extra_s) / CONTROL_DT)
    args.backflip_double_max_policy_steps = max(1, int(args.backflip_double_episode_s / POLICY_DT))
    args.backflip_double_max_steps = int(
        (args.backflip_double_episode_s + args.backflip_extra_s) / CONTROL_DT
    )
    args.turn_max_steps = int(args.turn_s / CONTROL_DT)
    args.turn_yaw_rad = math.radians(args.turn_yaw_deg)
    args.forward_ramp_steps = int(args.forward_ramp_s / CONTROL_DT)
    args.forward_max_steps = int(args.forward_s / CONTROL_DT) + args.forward_ramp_steps
    args.post_turn_settle_steps = int(args.post_turn_settle_s / CONTROL_DT)
    args.stop_max_steps = int(args.stop_s / CONTROL_DT)
    args.spring_jump_max_steps = int(args.spring_jump_s / CONTROL_DT)
    args.bridge_steps = int(args.bridge_s / CONTROL_DT)

    repo_root = Path(__file__).resolve().parents[2]
    script_dir = Path(__file__).resolve().parent

    import genesis as gs
    from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")
    device = gs.device

    handstand_policy = load_policy(
        resolve_path(args.handstand_model, repo_root), None, device, OBS_STAND
    )

    def resolve_skill_path(path_str: str) -> Path:
        for root in (script_dir, repo_root):
            candidate = resolve_path(path_str, root)
            if candidate.exists():
                return candidate
        return resolve_path(path_str, script_dir)

    cycle_ckpt = resolve_skill_path(args.legstand_cycle_ckpt)
    cycle_policy = load_policy(cycle_ckpt, None, device, OBS_CYCLE)
    backflip_path = resolve_skill_path(args.backflip_model)
    backflip_policy = load_policy(backflip_path, None, device, OBS_BACKFLIP)
    backflip_double_path = resolve_skill_path(args.backflip_double_model)
    backflip_double_policy = load_policy(backflip_double_path, None, device, OBS_BACKFLIP)
    turn_ckpt = resolve_skill_path(args.multivel_ckpt)
    turn_policy = load_policy(turn_ckpt, None, device, OBS_TURN)
    spring_path = resolve_skill_path(args.spring_jump_model)
    spring_policy = load_policy(spring_path, None, device, OBS_JUMP)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=CONTROL_DT, substeps=args.sim_substeps),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_self_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=80,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, -2.8, 1.3),
            camera_lookat=(0.3, 0.0, 0.45),
            camera_fov=45,
            max_FPS=args.fps,
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
        show_viewer=not args.headless,
    )
    scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    if args.model_format == "mjcf":
        robot = scene.add_entity(
            gs.morphs.MJCF(
                file=str(resolve_path(args.mjcf_model, repo_root)),
                pos=[0.0, 0.0, INIT_HEIGHT],
                quat=[1.0, 0.0, 0.0, 0.0],
            )
        )
    else:
        robot = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                merge_fixed_links=False,
                links_to_keep=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                pos=[0.0, 0.0, INIT_HEIGHT],
                quat=[1.0, 0.0, 0.0, 0.0],
            )
        )
    record_cam = None
    if args.headless:
        record_cam = scene.add_camera(
            res=(1280, 720),
            pos=(3.5, -2.8, 1.3),
            lookat=(0.3, 0.0, 0.45),
            fov=45,
        )
    scene.build(n_envs=1)
    if record_cam is not None:
        record_cam.follow_entity(robot, smoothing=0.9)

    isaac_dofs = [robot.get_joint(name).dof_start for name in JOINT_NAMES]
    genesis_dofs = [robot.get_joint(name).dof_start for name in GENESIS_JOINT_NAMES]

    quad_default = DEFAULT_QUAD_DOF.to(device)
    cycle_default = CYCLE_QUAD_DOF.to(device)
    backflip_default = BACKFLIP_QUAD_DOF.to(device)
    turn_default = MULTIVEL_QUAD_DOF.to(device)
    spring_default = SPRING_JUMP_DOF.to(device)
    hs_desired_delta = (HANDSTAND_DESIRED - DEFAULT_QUAD_DOF).to(device).unsqueeze(0)
    hs_joints = HANDSTAND_DESIRED.to(device)
    prone_joints = PRONE_FORWARD_DOF.to(device)
    low_joints = QUAD_LOW_DOF.to(device)
    quad_joints = DEFAULT_QUAD_DOF.to(device)

    rear_kp_boost = torch.ones((1, NUM_ACTIONS), dtype=torch.float32, device=device)
    rear_kp_boost[0, 6:] = args.recover_rear_kp_boost

    stand_kp = torch.full((1, NUM_ACTIONS), STAND_KP, device=device)
    stand_kd = torch.full((1, NUM_ACTIONS), STAND_KD * args.kd_scale, device=device)
    stand_tau = torch.full((1, NUM_ACTIONS), STAND_TAU, device=device)
    cycle_kp = torch.full((1, NUM_ACTIONS), CYCLE_KP, device=device)
    cycle_kd = torch.full((1, NUM_ACTIONS), CYCLE_KD, device=device)
    cycle_tau = torch.full((1, NUM_ACTIONS), CYCLE_TAU, device=device)
    flip_kp = torch.full((1, NUM_ACTIONS), BACKFLIP_KP, device=device)
    flip_kd = torch.full((1, NUM_ACTIONS), BACKFLIP_KD, device=device)
    flip_tau = torch.full((1, NUM_ACTIONS), BACKFLIP_TAU, device=device)
    turn_kp = torch.full((1, NUM_ACTIONS), TURN_KP, device=device)
    turn_kd = torch.full((1, NUM_ACTIONS), TURN_KD, device=device)
    turn_tau = torch.full((1, NUM_ACTIONS), TURN_TAU, device=device)
    jump_kp = torch.full((1, NUM_ACTIONS), JUMP_KP, device=device)
    jump_kd = torch.full((1, NUM_ACTIONS), JUMP_KD, device=device)
    jump_tau = torch.full((1, NUM_ACTIONS), JUMP_TAU, device=device)

    robot.set_dofs_position(quad_default.unsqueeze(0), dofs_idx_local=isaac_dofs, zero_velocity=True)
    robot.set_pos(torch.tensor([[0.0, 0.0, INIT_HEIGHT]], dtype=gs.tc_float, device=device), zero_velocity=True)
    robot.set_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=gs.tc_float, device=device), zero_velocity=True)

    zero_cmd = torch.zeros(3, dtype=torch.float32, device=device)
    turn_cmd = torch.tensor([0.0, 0.0, args.turn_ang_vel], dtype=torch.float32, device=device)
    forward_cmd = torch.tensor([args.forward_vx, 0.0, 0.0], dtype=torch.float32, device=device)
    rt = PhaseRuntime()
    rt.reset_stand_policy(device)
    rt.reset_genesis_policy(device)

    total_steps = int(
        args.handstand_max_steps
        + args.hold_handstand_steps
        + args.recover_quad_max
        + args.hold_quad_steps
        + args.cycle_max_steps
        + args.hold_quad_steps
        + args.backflip_max_steps
        + args.hold_quad_steps
        + args.turn_max_steps
        + args.post_turn_settle_steps
        + args.forward_max_steps
        + args.stop_max_steps
        + args.spring_jump_max_steps * 2
        + args.bridge_steps * 2
        + args.backflip_double_max_steps
        + 2000
    )
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = script_dir / video_path
    if record_cam is not None:
        record_cam.start_recording()

    last_policy_torques = torch.zeros((1, NUM_ACTIONS), device=device)
    last_control_dofs = isaac_dofs
    last_pos_target = None
    use_pos_control = False
    multivel_pd_ready = False
    flip_pd_ready = False
    anchor_pg = None
    anchor_prog = None
    anchor_ang = None

    with torch.no_grad():
        for step in range(total_steps):
            if rt.phase == Phase.DONE:
                break

            rt.phase_step += 1
            torques = last_policy_torques
            control_dofs = last_control_dofs
            pos_target = None
            use_pos_control = False

            if rt.phase in (Phase.HANDSTAND, Phase.HOLD_HANDSTAND):
                control_dofs = isaac_dofs
                torques, pg, stand_prog, ang = run_stand_phase(
                    rt,
                    handstand_policy,
                    robot,
                    isaac_dofs,
                    quad_default.unsqueeze(0),
                    hs_desired_delta,
                    zero_cmd,
                    device,
                    stand_kp,
                    stand_kd,
                    stand_tau,
                    step,
                    args,
                )
                last_policy_torques = torques
                anchor_pg, anchor_prog, anchor_ang = pg, stand_prog, ang
                stable = is_handstand_stable(pg, stand_prog, ang, args, rt.phase_step)
                if rt.phase == Phase.HANDSTAND:
                    rt.stable_steps = rt.stable_steps + 1 if stable else 0
                    if rt.stable_steps >= args.stable_steps_req:
                        advance_phase_with_robot(rt, Phase.HOLD_HANDSTAND, device, "handstand stable", robot)
                    elif rt.phase_step >= args.handstand_max_steps:
                        advance_phase_with_robot(rt, Phase.HOLD_HANDSTAND, device, "handstand timeout", robot)
                elif rt.phase_step >= args.hold_handstand_steps:
                    advance_phase_with_robot(rt, Phase.RECOVER_QUAD, device, "hold complete", robot)

            elif rt.phase == Phase.RECOVER_QUAD:
                control_dofs = isaac_dofs
                t = min(1.0, rt.phase_step / max(1, args.recover_descend_steps))
                joint_target = recovery_handstand_to_quad_joints(
                    t, hs_joints, prone_joints, low_joints, quad_joints
                )
                dof_pos = robot.get_dofs_position(isaac_dofs)
                dof_vel = robot.get_dofs_velocity(isaac_dofs)
                torques = run_scripted_recovery(
                    rt,
                    dof_pos,
                    dof_vel,
                    args,
                    joint_target.unsqueeze(0),
                    stand_kp,
                    stand_kd,
                    stand_tau,
                    kp_boost=rear_kp_boost,
                )
                last_policy_torques = torques
                base_quat = robot.get_quat()
                body_up = transform_by_quat(
                    torch.tensor([[0.0, 0.0, 1.0]], dtype=gs.tc_float, device=device),
                    base_quat,
                )
                levelness = float(torch.clamp(body_up[0, 2], 0.0, 1.0))
                start_ramp = smoothstep(min(t / 0.15, 1.0))
                att_fade = (1.0 - levelness) * start_ramp
                apply_body_level_torque(robot, scene, device, args, att_fade)

                inv_base_quat = inv_quat(robot.get_quat())
                pg = transform_by_quat(
                    torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=device),
                    inv_base_quat,
                )
                ang = transform_by_quat(robot.get_ang(), inv_base_quat)
                quad_ready = is_transition_ready(pg, dof_pos, quad_default, ang, args)
                if quad_ready and t > 0.85:
                    rt.stable_steps += 1
                else:
                    rt.stable_steps = 0
                if rt.stable_steps >= args.transition_steps_req:
                    advance_phase_with_robot(rt, Phase.HOLD_QUAD, device, "descended to quad", robot)
                elif rt.phase_step >= args.recover_quad_max:
                    advance_phase_with_robot(rt, Phase.HOLD_QUAD, device, "descend timeout", robot)

            elif rt.phase == Phase.HOLD_QUAD:
                # Brief handoff toward cycle's default FR/FL/RR/RL stance (not a long Isaac freeze).
                control_dofs = genesis_dofs
                torques, pg, ang, dof_pos = run_hold_pose(
                    rt,
                    robot,
                    genesis_dofs,
                    cycle_default,
                    cycle_default.unsqueeze(0),
                    device,
                    cycle_kp,
                    cycle_kd,
                    cycle_tau,
                    zero_cmd,
                )
                last_policy_torques = torques
                if is_transition_ready(pg, dof_pos, cycle_default, ang, args):
                    rt.stable_steps += 1
                else:
                    rt.stable_steps = 0
                if rt.stable_steps >= args.transition_steps_req:
                    advance_phase_with_robot(
                        rt,
                        Phase.LEGSTAND_CYCLE,
                        device,
                        "handoff to cycle",
                        robot,
                        warm_start_dofs=genesis_dofs,
                        warm_start_default=cycle_default.unsqueeze(0),
                        warm_start_scale=CYCLE_ACTION_SCALE,
                    )
                elif rt.phase_step >= args.hold_quad_steps:
                    advance_phase_with_robot(
                        rt,
                        Phase.LEGSTAND_CYCLE,
                        device,
                        "handoff timeout",
                        robot,
                        warm_start_dofs=genesis_dofs,
                        warm_start_default=cycle_default.unsqueeze(0),
                        warm_start_scale=CYCLE_ACTION_SCALE,
                    )

            elif rt.phase == Phase.LEGSTAND_CYCLE:
                control_dofs = genesis_dofs
                torques, pg, ang, dof_pos, stand_cmd, _ = run_legstand_cycle_phase(
                    rt,
                    cycle_policy,
                    robot,
                    genesis_dofs,
                    cycle_default.unsqueeze(0),
                    zero_cmd,
                    device,
                    cycle_kp,
                    cycle_kd,
                    cycle_tau,
                    step,
                    args.cycle_hold_end_s,
                )
                last_policy_torques = torques
                # Exit recover ASAP once roughly upright — do NOT wait for joint match / full 30s.
                in_recover = stand_cmd > 0.5
                if in_recover and is_cycle_landed(robot, pg, ang, args):
                    rt.stable_steps += 1
                else:
                    rt.stable_steps = 0
                leave_cycle = False
                leave_reason = ""
                if in_recover and rt.stable_steps >= args.transition_steps_req:
                    leave_cycle = True
                    leave_reason = "cycle landed"
                elif in_recover and rt.phase_step >= int(
                    (args.cycle_hold_end_s + args.cycle_recover_max_s) / CONTROL_DT
                ):
                    leave_cycle = True
                    leave_reason = f"cycle recover cap ({args.cycle_recover_max_s:.1f}s)"
                elif rt.phase_step >= args.cycle_max_steps:
                    leave_cycle = True
                    leave_reason = "cycle timeout"
                if leave_cycle:
                    advance_phase_with_robot(rt, Phase.HOLD_QUAD_2, device, leave_reason, robot)

            elif rt.phase == Phase.HOLD_QUAD_2:
                # Micro handoff (~hold_quad_s) toward backflip default — exit on first calm frame or cap.
                control_dofs = genesis_dofs
                torques, pg, ang, dof_pos = run_hold_pose(
                    rt,
                    robot,
                    genesis_dofs,
                    backflip_default,
                    backflip_default.unsqueeze(0),
                    device,
                    flip_kp,
                    flip_kd,
                    flip_tau,
                    zero_cmd,
                )
                last_policy_torques = torques
                ready = is_cycle_landed(robot, pg, ang, args) or is_transition_ready(
                    pg, dof_pos, backflip_default, ang, args
                )
                if ready:
                    rt.stable_steps += 1
                else:
                    rt.stable_steps = 0
                if rt.stable_steps >= args.transition_steps_req or rt.phase_step >= args.hold_quad_steps:
                    reason = "handoff to backflip" if rt.stable_steps >= args.transition_steps_req else "handoff timeout"
                    advance_phase_with_robot(
                        rt,
                        Phase.BACKFLIP,
                        device,
                        reason,
                        robot,
                        warm_start_dofs=genesis_dofs,
                        warm_start_default=backflip_default.unsqueeze(0),
                        warm_start_scale=BACKFLIP_ACTION_SCALE,
                    )

            elif rt.phase == Phase.BACKFLIP:
                control_dofs = genesis_dofs
                out, pg, ang, dof_pos, use_pos_control = run_backflip_phase(
                    rt,
                    backflip_policy,
                    robot,
                    genesis_dofs,
                    backflip_default.unsqueeze(0),
                    device,
                    flip_kp,
                    flip_kd,
                    flip_tau,
                    step,
                    args.backflip_max_policy_steps,
                    use_position=True,
                )
                if use_pos_control:
                    if not flip_pd_ready:
                        robot.set_dofs_kp([BACKFLIP_KP] * NUM_ACTIONS, genesis_dofs)
                        robot.set_dofs_kv([BACKFLIP_KD] * NUM_ACTIONS, genesis_dofs)
                        flip_pd_ready = True
                    pos_target = out
                else:
                    torques = out
                    last_policy_torques = torques
                if rt.phase_step >= args.backflip_max_steps:
                    advance_phase_with_robot(rt, Phase.HOLD_QUAD_3, device, "backflip complete", robot)

            elif rt.phase == Phase.HOLD_QUAD_3:
                # Brief settle toward multivel default before in-place turn.
                control_dofs = genesis_dofs
                torques, pg, ang, dof_pos = run_hold_pose(
                    rt,
                    robot,
                    genesis_dofs,
                    turn_default,
                    turn_default.unsqueeze(0),
                    device,
                    turn_kp,
                    turn_kd,
                    turn_tau,
                    zero_cmd,
                )
                last_policy_torques = torques
                ready = is_cycle_landed(robot, pg, ang, args) or is_transition_ready(
                    pg, dof_pos, turn_default, ang, args
                )
                if ready:
                    rt.stable_steps += 1
                else:
                    rt.stable_steps = 0
                if rt.stable_steps >= args.transition_steps_req or rt.phase_step >= args.hold_quad_steps:
                    reason = "handoff to turn" if rt.stable_steps >= args.transition_steps_req else "handoff timeout"
                    advance_phase_with_robot(
                        rt,
                        Phase.TURN_INPLACE,
                        device,
                        reason,
                        robot,
                        warm_start_dofs=genesis_dofs,
                        warm_start_default=turn_default.unsqueeze(0),
                        warm_start_scale=TURN_ACTION_SCALE,
                    )

            elif rt.phase == Phase.TURN_INPLACE:
                control_dofs = genesis_dofs
                if not multivel_pd_ready:
                    robot.set_dofs_kp([TURN_KP] * NUM_ACTIONS, genesis_dofs)
                    robot.set_dofs_kv([TURN_KD] * NUM_ACTIONS, genesis_dofs)
                    multivel_pd_ready = True
                if rt.turn_yaw0 is None:
                    rt.turn_yaw0 = yaw_from_quat(robot.get_quat())
                # Taper command near the target so we don't overshoot past 90°.
                yaw_now = yaw_from_quat(robot.get_quat())
                # Left turn (+ω) → positive yaw delta in body/world convention used here.
                signed_delta = wrap_to_pi(yaw_now - rt.turn_yaw0)
                progress = abs(signed_delta) / max(args.turn_yaw_rad, 1e-6)
                remain = max(0.0, 1.0 - progress)
                cmd_scale = 1.0 if remain > 0.25 else max(0.25, remain / 0.25)
                turn_cmd_scaled = turn_cmd * cmd_scale
                out, pg, ang, dof_pos, use_pos_control = run_turn_phase(
                    rt,
                    turn_policy,
                    robot,
                    genesis_dofs,
                    turn_default.unsqueeze(0),
                    turn_cmd_scaled,
                    device,
                    turn_kp,
                    turn_kd,
                    turn_tau,
                    step,
                    use_position=True,
                )
                if use_pos_control:
                    pos_target = out
                else:
                    torques = out
                    last_policy_torques = torques
                # 2° tolerance: stop as soon as we are essentially at the target yaw.
                reached = abs(signed_delta) >= max(0.0, args.turn_yaw_rad - math.radians(2.0))
                if reached:
                    advance_phase_with_robot(
                        rt,
                        Phase.SETTLE_AFTER_TURN,
                        device,
                        f"turned {math.degrees(abs(signed_delta)):.0f}°",
                        robot,
                    )
                elif rt.phase_step >= args.turn_max_steps:
                    advance_phase_with_robot(
                        rt,
                        Phase.SETTLE_AFTER_TURN,
                        device,
                        f"turn timeout ({math.degrees(abs(signed_delta)):.0f}°)",
                        robot,
                    )

            elif rt.phase == Phase.SETTLE_AFTER_TURN:
                # Zero command: let multivel recover a standing gait before accelerating.
                control_dofs = genesis_dofs
                out, pg, ang, dof_pos, use_pos_control = run_turn_phase(
                    rt,
                    turn_policy,
                    robot,
                    genesis_dofs,
                    turn_default.unsqueeze(0),
                    zero_cmd,
                    device,
                    turn_kp,
                    turn_kd,
                    turn_tau,
                    step,
                    use_position=True,
                )
                if use_pos_control:
                    pos_target = out
                else:
                    torques = out
                    last_policy_torques = torques
                if rt.phase_step >= args.post_turn_settle_steps:
                    advance_phase_with_robot(rt, Phase.FORWARD, device, "settled after turn", robot)

            elif rt.phase == Phase.FORWARD:
                # Ramp vx 0→target, then hold for forward_s (avoids step-command faceplant).
                control_dofs = genesis_dofs
                if rt.phase_step <= args.forward_ramp_steps:
                    alpha = smoothstep(rt.phase_step / max(1, args.forward_ramp_steps))
                else:
                    alpha = 1.0
                fwd = forward_cmd.clone()
                fwd[0] = args.forward_vx * alpha
                out, pg, ang, dof_pos, use_pos_control = run_turn_phase(
                    rt,
                    turn_policy,
                    robot,
                    genesis_dofs,
                    turn_default.unsqueeze(0),
                    fwd,
                    device,
                    turn_kp,
                    turn_kd,
                    turn_tau,
                    step,
                    use_position=True,
                )
                if use_pos_control:
                    pos_target = out
                else:
                    torques = out
                    last_policy_torques = torques
                if rt.phase_step >= args.forward_max_steps:
                    advance_phase_with_robot(rt, Phase.STOP, device, "forward complete", robot)

            elif rt.phase == Phase.STOP:
                # Zero command so the multivel policy brakes to a standstill.
                control_dofs = genesis_dofs
                out, pg, ang, dof_pos, use_pos_control = run_turn_phase(
                    rt,
                    turn_policy,
                    robot,
                    genesis_dofs,
                    turn_default.unsqueeze(0),
                    zero_cmd,
                    device,
                    turn_kp,
                    turn_kd,
                    turn_tau,
                    step,
                    use_position=True,
                )
                if use_pos_control:
                    pos_target = out
                else:
                    torques = out
                    last_policy_torques = torques
                if rt.phase_step >= args.stop_max_steps:
                    advance_phase_with_robot(rt, Phase.SPRING_JUMP, device, "stopped → spring jump", robot)
                    prefill_jump_history(
                        rt,
                        robot,
                        isaac_dofs,
                        spring_default,
                        device,
                        quat_to_xyz,
                        args.jump_distance,
                    )

            elif rt.phase == Phase.SPRING_JUMP:
                # Mirror go2_eval_spring_jump_gym_policy: FL/FR/RL/RR, 47×10 stack, torque PD.
                control_dofs = isaac_dofs
                use_pos_control = False
                torques, dof_pos, dof_vel = run_spring_jump_phase(
                    rt,
                    spring_policy,
                    robot,
                    isaac_dofs,
                    spring_default.unsqueeze(0),
                    device,
                    jump_kp,
                    jump_kd,
                    jump_tau,
                    step,
                    args.jump_distance,
                    args.jump_frame,
                    args.jump_startup_steps,
                    quat_to_xyz,
                )
                last_policy_torques = torques
                # End soon after landing instead of idling through the full episode.
                inv_q = inv_quat(robot.get_quat())
                pg = transform_by_quat(
                    torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=device),
                    inv_q,
                )
                ang = transform_by_quat(robot.get_ang(), inv_q)
                after_takeoff = rt.jump_policy_step >= args.jump_frame + 30  # ~0.6s after jump flag
                if after_takeoff and (
                    is_cycle_landed(robot, pg, ang, args)
                    or is_transition_ready(pg, dof_pos, spring_default, ang, args)
                ):
                    rt.stable_steps += 1
                else:
                    rt.stable_steps = 0
                jump_done = (after_takeoff and rt.stable_steps >= args.transition_steps_req) or (
                    rt.phase_step >= args.spring_jump_max_steps
                )
                if jump_done:
                    rt.jump_count += 1
                    reason = (
                        f"spring jump {rt.jump_count}/2 landed"
                        if after_takeoff and rt.stable_steps >= args.transition_steps_req
                        else f"spring jump {rt.jump_count}/2 timeout"
                    )
                    advance_phase_with_robot(rt, Phase.HOLD_BRIDGE, device, reason, robot)

            elif rt.phase == Phase.HOLD_BRIDGE:
                # Brief settle (bridge_s ≈ 1.5s) between jumps / before finale backflips.
                if rt.jump_count < 2:
                    control_dofs = isaac_dofs
                    hold_target = spring_default
                    hold_kp, hold_kd, hold_tau = jump_kp, jump_kd, jump_tau
                    default_ref = spring_default.unsqueeze(0)
                else:
                    control_dofs = genesis_dofs
                    hold_target = backflip_default
                    hold_kp, hold_kd, hold_tau = flip_kp, flip_kd, flip_tau
                    default_ref = backflip_default.unsqueeze(0)

                use_pos_control = False
                torques, pg, ang, dof_pos = run_hold_pose(
                    rt,
                    robot,
                    control_dofs,
                    hold_target,
                    default_ref,
                    device,
                    hold_kp,
                    hold_kd,
                    hold_tau,
                    zero_cmd,
                )
                last_policy_torques = torques
                if rt.phase_step >= args.bridge_steps:
                    if rt.jump_count < 2:
                        advance_phase_with_robot(
                            rt, Phase.SPRING_JUMP, device, f"start spring jump {rt.jump_count + 1}/2", robot
                        )
                        prefill_jump_history(
                            rt,
                            robot,
                            isaac_dofs,
                            spring_default,
                            device,
                            quat_to_xyz,
                            args.jump_distance,
                        )
                    else:
                        advance_phase_with_robot(
                            rt,
                            Phase.BACKFLIP_FINALE,
                            device,
                            "start backflip_double",
                            robot,
                            warm_start_dofs=genesis_dofs,
                            warm_start_default=backflip_default.unsqueeze(0),
                            warm_start_scale=BACKFLIP_ACTION_SCALE,
                        )

            elif rt.phase == Phase.BACKFLIP_FINALE:
                # One continuous double backflip (double.pt, episode_length_s=3).
                control_dofs = genesis_dofs
                out, pg, ang, dof_pos, use_pos_control = run_backflip_phase(
                    rt,
                    backflip_double_policy,
                    robot,
                    genesis_dofs,
                    backflip_default.unsqueeze(0),
                    device,
                    flip_kp,
                    flip_kd,
                    flip_tau,
                    step,
                    args.backflip_double_max_policy_steps,
                    use_position=True,
                )
                if use_pos_control:
                    if not flip_pd_ready:
                        robot.set_dofs_kp([BACKFLIP_KP] * NUM_ACTIONS, genesis_dofs)
                        robot.set_dofs_kv([BACKFLIP_KD] * NUM_ACTIONS, genesis_dofs)
                        flip_pd_ready = True
                    pos_target = out
                else:
                    torques = out
                    last_policy_torques = torques
                if rt.phase_step >= args.backflip_double_max_steps:
                    advance_phase_with_robot(rt, Phase.DONE, device, "backflip_double complete", robot)

            last_control_dofs = control_dofs

            if should_apply_anchor(rt, rt.phase, args, anchor_pg, anchor_prog, anchor_ang):
                apply_anchor_drift_control(robot, scene, rt, args, device)

            if use_pos_control and pos_target is not None:
                robot.control_dofs_position(pos_target, control_dofs)
                last_pos_target = pos_target
            else:
                robot.control_dofs_force(torques, control_dofs)
            scene.step()
            if record_cam is not None and step % max(1, int(1.0 / (CONTROL_DT * args.fps))) == 0:
                record_cam.update_following()
                record_cam.render()

    if record_cam is not None:
        record_cam.stop_recording(save_to_filename=str(video_path), fps=args.fps)
        print(f"视频已保存: {video_path}")
    else:
        input("Genesis combo 结束，按 Enter 退出...")


if __name__ == "__main__":
    main()
