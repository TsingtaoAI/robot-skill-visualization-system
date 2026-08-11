"""
go2_env_multivel.py
===================
多速度指令训练环境。

相比 go2_env_dr.py 的改动：
  1. 速度指令范围扩展为全向：
       lin_vel_x ∈ [-1.0, 1.0]  前进/后退
       lin_vel_y ∈ [-0.5, 0.5]  左右侧移
       ang_vel   ∈ [-1.0, 1.0]  左右转向
  2. 去掉 alive 奖励（会导致趴着不动的局部最优）
  3. base_height 系数 -1.0（原 baseline 里的 -50 是 bug）
  4. termination 角度放宽到 25°（原 10° 在 DR 扰动下太严格）
  5. DR 和 B-v2 保持一致（已验证可用）
"""

import math
import torch
from tensordict import TensorDict
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(
        size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device
    ) + lower


class Go2EnvMultiVel:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        dr_cfg=None,
        show_viewer=True,
        num_obs=45,
    ):
        self.num_envs    = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg         = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device      = gs.device

        self.simulate_action_latency = True
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg     = env_cfg
        self.obs_cfg     = obs_cfg
        self.reward_cfg  = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales  = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.enable_dr = dr_cfg is not None
        self.dr_cfg    = dr_cfg if dr_cfg is not None else {}
        self._step_count = 0
        # 交互导航 / Viser 遥控时冻结速度指令，避免 step() 内定时重采样覆盖
        self.freeze_commands = bool(self.env_cfg.get("freeze_commands", False))
        # 可选导航终点（世界系 xy），供 debug / Viser 同步
        self.nav_goal_xy = None

        # ── 场景 ──────────────────────────────────────────────────
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=max(64, len(self.env_cfg.get("demo_obstacles", [])) * 2),
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )

        # ── 平地 ──────────────────────────────────────────────────
        self.scene.add_entity(
            gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
        )

        # ── 演示障碍物 ─────────────────────────────────────────────
        # eval 脚本可通过 env_cfg["demo_obstacles"] 注入棋盘格/围栏等障碍，
        # 保持训练和普通评估默认仍然是纯平地。
        self.demo_obstacles = []
        for obstacle in self.env_cfg.get("demo_obstacles", []):
            self.demo_obstacles.append(
                self.scene.add_entity(
                    gs.morphs.Box(
                        pos=obstacle["pos"],
                        size=obstacle["size"],
                        fixed=True,
                    )
                )
            )

        # ── 机器人 ────────────────────────────────────────────────
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.env_cfg["base_init_pos"],
                quat=self.env_cfg["base_init_quat"],
            ),
        )

        # ── 官方 Lidar（eval 绕障演示用，须在 build 前挂载）────────────
        self.lidar = None
        if self.env_cfg.get("use_official_lidar"):
            lidar_cfg = self.env_cfg.get("lidar_cfg", {})
            pattern = gs.sensors.SphericalPattern(
                fov=lidar_cfg.get("fov", (360.0, 0.0)),
                n_points=lidar_cfg.get("n_points", (181, 1)),
            )
            self.lidar = self.scene.add_sensor(
                gs.sensors.Lidar(
                    pattern=pattern,
                    entity_idx=self.robot.idx,
                    pos_offset=lidar_cfg.get("pos_offset", (0.3, 0.0, 0.1)),
                    euler_offset=lidar_cfg.get("euler_offset", (0.0, 0.0, 0.0)),
                    max_range=lidar_cfg.get("max_range", 3.0),
                    return_world_frame=True,
                    draw_debug=lidar_cfg.get("draw_debug", False),
                )
            )

        self.scene.build(n_envs=num_envs)

        # ── 关节索引 ──────────────────────────────────────────────
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int, device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # ── 重力向量 ──────────────────────────────────────────────
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        # ── 初始状态 ──────────────────────────────────────────────
        self.init_base_pos  = torch.tensor(self.env_cfg["base_init_pos"],  dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float, device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # ── buffers ───────────────────────────────────────────────
        self.base_lin_vel       = torch.zeros((self.num_envs, 3),  dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel       = torch.zeros((self.num_envs, 3),  dtype=gs.tc_float, device=gs.device)
        self.projected_gravity  = torch.zeros((self.num_envs, 3),  dtype=gs.tc_float, device=gs.device)
        self.rew_buf            = torch.zeros((self.num_envs,),    dtype=gs.tc_float, device=gs.device)
        self.reset_buf          = torch.ones ((self.num_envs,),    dtype=gs.tc_bool,  device=gs.device)
        self.episode_length_buf = torch.zeros((self.num_envs,),    dtype=gs.tc_int,   device=gs.device)
        self.commands           = torch.zeros((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale     = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device, dtype=gs.tc_float,
        )

        # 速度指令范围（支持全向）
        lx_lo, lx_hi = self.command_cfg["lin_vel_x_range"]
        ly_lo, ly_hi = self.command_cfg["lin_vel_y_range"]
        av_lo, av_hi = self.command_cfg["ang_vel_range"]
        self._cmd_lo = torch.tensor([lx_lo, ly_lo, av_lo], dtype=gs.tc_float, device=gs.device)
        self._cmd_hi = torch.tensor([lx_hi, ly_hi, av_hi], dtype=gs.tc_float, device=gs.device)

        self.actions      = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos      = torch.zeros_like(self.actions)
        self.dof_vel      = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos     = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat    = torch.zeros((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler   = torch.zeros((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float, device=gs.device,
        )
        self.extras = dict()

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        self.reset()

    # ── Domain Randomization ──────────────────────────────────────

    def _randomize_physics(self, envs_idx):
        if not self.enable_dr:
            return
        idx = (torch.where(envs_idx)[0]
               if envs_idx is not None
               else torch.arange(self.num_envs, device=gs.device))
        if idx.numel() == 0:
            return
        n = idx.numel()

        # 1. 摩擦系数（批量）
        fr_lo = self.dr_cfg.get("friction_ratio_range", [0.7, 1.3])[0]
        fr_hi = self.dr_cfg.get("friction_ratio_range", [0.7, 1.3])[1]
        num_links = len(self.robot.links)
        friction_ratios = torch.empty(n, num_links, dtype=gs.tc_float, device=gs.device).uniform_(fr_lo, fr_hi)
        self.robot.set_friction_ratio(friction_ratios, envs_idx=idx)

        # 2. 关节摩擦损失（逐env，Genesis 0.4.6 API限制）
        df_lo = self.dr_cfg.get("dof_frictionloss_range", [0.0, 0.08])[0]
        df_hi = self.dr_cfg.get("dof_frictionloss_range", [0.0, 0.08])[1]
        dof_fl_all = torch.empty(n, len(self.motors_dof_idx), dtype=gs.tc_float, device=gs.device).uniform_(df_lo, df_hi)
        for ei, i in enumerate(idx):
            env_idx_t = torch.tensor([i.item()], dtype=gs.tc_int, device=gs.device)
            for ji, dof_idx in enumerate(self.motors_dof_idx):
                self.robot.set_dofs_frictionloss(
                    dof_fl_all[ei, ji:ji+1], dof_idx, envs_idx=env_idx_t
                )

    def _apply_push(self):
        if not self.enable_dr:
            return
        interval = self.dr_cfg.get("push_interval_steps", 500)
        if self._step_count % interval != 0:
            return
        vmax = self.dr_cfg.get("push_vel_max", 0.2)
        cur_vel = self.robot.get_dofs_velocity()
        cur_vel[:, 0] += torch.empty(self.num_envs, dtype=gs.tc_float, device=gs.device).uniform_(-vmax, vmax)
        cur_vel[:, 1] += torch.empty(self.num_envs, dtype=gs.tc_float, device=gs.device).uniform_(-vmax, vmax)
        self.robot.set_dofs_velocity(cur_vel)

    # ── 核心接口 ──────────────────────────────────────────────────

    def _resample_commands(self, envs_idx):
        """全向速度指令随机采样，小速度截断为0（dead zone，避免策略学到微小抖动）"""
        new_cmd = gs_rand(self._cmd_lo, self._cmd_hi, (self.num_envs,))

        # dead zone：|vel| < 0.1 时截断为 0，让策略学会"停止"这个技能
        dead = 0.1
        new_cmd[:, 0] = torch.where(new_cmd[:, 0].abs() < dead, torch.zeros_like(new_cmd[:, 0]), new_cmd[:, 0])
        new_cmd[:, 1] = torch.where(new_cmd[:, 1].abs() < dead, torch.zeros_like(new_cmd[:, 1]), new_cmd[:, 1])
        new_cmd[:, 2] = torch.where(new_cmd[:, 2].abs() < dead, torch.zeros_like(new_cmd[:, 2]), new_cmd[:, 2])

        if envs_idx is None:
            self.commands.copy_(new_cmd)
        else:
            torch.where(envs_idx[:, None], new_cmd, self.commands, out=self.commands)

    def step(self, actions):
        self._step_count += 1
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))

        self._apply_push()
        self.scene.step()

        self.episode_length_buf += 1
        self.base_pos   = self.robot.get_pos()
        self.base_quat  = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat          = inv_quat(self.base_quat)
        self.base_lin_vel      = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel      = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        if not self.freeze_commands:
            self._resample_commands(
                self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0
            )

        self.reset_buf  = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        self.reset_buf |= self.scene.rigid_solver.get_error_envs_mask()

        self.extras["time_outs"] = (self.episode_length_buf > self.max_episode_length).to(dtype=gs.tc_float)

        self._reset_idx(self.reset_buf)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)

        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def _reset_idx(self, envs_idx=None):
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
        else:
            torch.where(envs_idx[:, None], self.init_base_pos,         self.base_pos,         out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat,        self.base_quat,        out=self.base_quat)
            torch.where(envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity)
            torch.where(envs_idx[:, None], self.init_dof_pos,          self.dof_pos,          out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)

        self._randomize_physics(envs_idx)

        n_envs = envs_idx.sum() if envs_idx is not None else self.num_envs
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if envs_idx is None:
                mean = value.mean()
            else:
                mean = torch.where(n_envs > 0, value[envs_idx].sum() / n_envs, 0.0)
            self.extras["episode"]["rew_" + key] = mean / self.env_cfg["episode_length_s"]
            if envs_idx is None:
                value.zero_()
            else:
                value.masked_fill_(envs_idx, 0.0)

        if not self.freeze_commands:
            self._resample_commands(envs_idx)

    def set_nav_goal(self, xy, z=0.08):
        """设置交互导航终点（世界系）；xy=None 表示清除。"""
        if xy is None:
            self.nav_goal_xy = None
            return
        self.nav_goal_xy = (float(xy[0]), float(xy[1]), float(z))

    def _update_observation(self):
        self.obs_buf = torch.concatenate(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )  # 45维，与原版完全兼容

    def reset(self):
        self._reset_idx()
        self._update_observation()
        return self.get_observations()

    # ── 奖励函数 ──────────────────────────────────────────────────

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """惩罚不必要的横滚/俯仰角速度，鼓励平稳行走"""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)