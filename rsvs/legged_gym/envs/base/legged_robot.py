from __future__ import annotations

from legged_gym import *
from time import time
import numpy as np
import os

import torch
from torch import Tensor
from typing import Tuple, Dict, List, Callable, Optional, Union, Any

from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math_utils import wrap_to_pi, torch_rand_float, quat_apply, quat_from_euler_xyz
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

# Type aliases for common tensor types
ObsBuf = Tensor  # Shape: (num_envs, obs_dim)
Action = Tensor  # Shape: (num_envs, num_actions)
Reward = Tensor  # Shape: (num_envs,)
EnvIds = Tensor  # Shape: (num_reset_envs,) - integer tensor of environment indices

class LeggedRobot(BaseTask):
    def __init__(
        self,
        cfg: LeggedRobotCfg,
        sim_params: dict[str, Any],
        sim_device: str | int,
        headless: bool
    ) -> None:
        """Initialize the legged robot environment.
        
        Parses the configuration, creates the simulation environment (including terrain
        and robot instances), and initializes PyTorch buffers for training.
        
        Args:
            cfg: Environment configuration containing robot, terrain, control, and
                reward parameters.
            sim_params: Dictionary of simulation parameters passed to the simulator
                backend (IsaacGym, Genesis, or IsaacSim).
            sim_device: Device for running the simulation. Can be 'cuda', 'cpu',
                or a device ID integer.
            headless: If True, run without rendering for faster training.
        
        Example:
            >>> cfg = GO2Cfg()
            >>> sim_params = {"dt": 0.005, "substeps": 1}
            >>> env = LeggedRobot(cfg, sim_params, "cuda:0", headless=True)
        """
        # === CONFIG VALIDATION ===
        # Validate config structure - help users debug configuration issues early
        assert hasattr(cfg, 'env'), (
            f"Config missing required 'env' section. "
            f"Check that cfg is a valid LeggedRobotCfg instance. "
            f"Got type: {type(cfg).__name__}"
        )
        assert hasattr(cfg.env, 'num_observations'), (
            f"Config.env missing 'num_observations'. "
            f"Required for observation buffer allocation. "
            f"Available env attributes: {dir(cfg.env)}"
        )
        assert hasattr(cfg.env, 'num_actions'), (
            f"Config.env missing 'num_actions'. "
            f"Required for action buffer allocation. "
            f"Available env attributes: {dir(cfg.env)}"
        )
        assert cfg.env.num_observations > 0, (
            f"Config.env.num_observations must be positive, got {cfg.env.num_observations}. "
            f"Update your config to specify a valid observation dimension."
        )
        assert cfg.env.num_actions > 0, (
            f"Config.env.num_actions must be positive, got {cfg.env.num_actions}. "
            f"Update your config to specify a valid action dimension."
        )
        assert hasattr(cfg, 'normalization'), (
            f"Config missing 'normalization' section. "
            f"Required for observation/action scaling. "
            f"Check cfg.normalization configuration."
        )
        assert hasattr(cfg.normalization, 'clip_observations'), (
            f"Config.normalization missing 'clip_observations'. "
            f"Required for observation clipping. "
            f"Add: cfg.normalization.clip_observations = 100.0"
        )
        assert hasattr(cfg.normalization, 'clip_actions'), (
            f"Config.normalization missing 'clip_actions'. "
            f"Required for action clipping. "
            f"Add: cfg.normalization.clip_actions = 100.0"
        )
        
        self.cfg: LeggedRobotCfg = cfg
        self.init_done: bool = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, sim_device, headless)
        
        # === SIMULATOR INITIALIZATION VALIDATION ===
        # Ensure simulator was initialized correctly
        assert hasattr(self, 'num_envs'), (
            "Simulator initialization failed: 'num_envs' not set. "
            "Check that BaseTask.__init__() was called and simulator created environments."
        )
        assert self.num_envs > 0, (
            f"Invalid num_envs: {self.num_envs}. "
            f"Simulator must create at least one environment."
        )
        assert hasattr(self, 'num_actions'), (
            f"'num_actions' not set. "
            f"Expected cfg.env.num_actions={cfg.env.num_actions}. "
            f"Check simulator initialization."
        )
        assert hasattr(self, 'num_obs'), (
            f"'num_obs' not set. "
            f"Expected cfg.env.num_observations={cfg.env.num_observations}. "
            f"Check simulator initialization."
        )
        
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions: Action) -> Tuple[ObsBuf, ObsBuf | None, Reward, Tensor, Dict[str, Any]]:
        """Execute one simulation step with the given actions.
        
        Applies actions to the robots, advances the physics simulation, then processes
        observations, rewards, and termination states.
        
        Args:
            actions: Action tensor of shape (num_envs, num_actions) containing
                target joint positions or torques depending on control mode.
        
        Returns:
            Tuple containing:
                - obs_buf: Observation buffer of shape (num_envs, obs_dim).
                - privileged_obs_buf: Privileged observations of shape (num_envs, privileged_obs_dim)
                    or None if not using asymmetric actor-critic.
                - rew_buf: Reward buffer of shape (num_envs,).
                - reset_buf: Reset flags of shape (num_envs,) indicating which environments
                    need reset.
                - extras: Dictionary of additional information including episode statistics.
        
        Example:
            >>> actions = torch.zeros(4096, 12, device="cuda")  # 12 DOFs for Go2
            >>> obs, priv_obs, rew, done, info = env.step(actions)
        """
        # === ACTION SHAPE VALIDATION ===
        expected_shape = (self.num_envs, self.num_actions)
        assert actions.shape == expected_shape, (
            f"Action shape mismatch: expected {expected_shape}, got {actions.shape}. "
            f"Actions must have shape (num_envs={self.num_envs}, num_actions={self.num_actions}). "
            f"Check your policy output or action preprocessing."
        )
        assert actions.dtype == torch.float32, (
            f"Actions must be float32, got {actions.dtype}. "
            f"Convert with: actions = actions.float()"
        )
        
        actions = self._pre_sim_step(actions)
        self.simulator.step(actions)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self) -> None:
        """Process environment state after physics step.
        
        Called after each simulation step to update episode counters, check termination
        conditions, compute rewards, reset terminated environments, and compute new
        observations. This method orchestrates the RL training loop logic.
        
        The execution order is:
            1. Update episode and step counters
            2. Process simulator post-physics callbacks
            3. Run custom callbacks (_post_physics_step_callback)
            4. Check termination conditions
            5. Compute rewards
            6. Reset terminated environments
            7. Update sensors
            8. Compute observations
            9. Draw debug visualizations (if enabled)
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.simulator.post_physics_step()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.simulator.update_sensors()
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        if self.debug:
            self.simulator.draw_debug_vis()

    def check_termination(self) -> None:
        """Check termination conditions and update reset buffer.
        
        Evaluates three termination conditions:
            1. Contact termination: Body contacts with termination bodies exceed threshold.
            2. Orientation termination: Projected gravity exceeds maximum allowed value.
            3. Timeout termination: Episode exceeds maximum episode length.
        
        Updates the following buffers:
            - fail_buf: Tracks consecutive failures for graceful termination.
            - time_out_buf: Indicates episodes that timed out (not actual failures).
            - reset_buf: Indicates environments needing reset.
        """
        # if the dim of link_contact_forces is 4, then it has history (IsaacLab). shape [N, T, B, 3] (N: num_envs, T: history length, B: number of links with contact sensors)
        if len(self.simulator.link_contact_forces.shape) == 4:
            self.terminated_bodies_force_norm = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.termination_contact_indices, :], dim=-1), dim=1)[0]
            self.penalized_bodies_force_norm = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.penalized_contact_indices, :], dim=-1), dim=1)[0]
            self.feet_force_norm = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.feet_contact_indices, :], dim=-1), dim=1)[0]
            self.feet_max_force_z = torch.max(self.simulator.link_contact_forces[:, :, self.simulator.feet_contact_indices, 2], dim=1)[0]
        else:
            self.terminated_bodies_force_norm = torch.norm(self.simulator.link_contact_forces[:, self.simulator.termination_contact_indices, :], dim=-1)
            self.penalized_bodies_force_norm = torch.norm(self.simulator.link_contact_forces[:, self.simulator.penalized_contact_indices, :], dim=-1)
            self.feet_force_norm = torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, :], dim=-1)
            self.feet_max_force_z = self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2]
        fail_buf = torch.any(self.terminated_bodies_force_norm > 10.0, dim=1)
        # print(f"contact termination: {fail_buf}")
        fail_buf |= self.simulator.projected_gravity[:, 2] > self.cfg.env.max_projected_gravity
        # print(f"gravity termination: {self.simulator.projected_gravity[:, 2] > self.cfg.env.max_projected_gravity}")
        self.fail_buf += fail_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        # print(f"time out: {self.time_out_buf}")
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
        )

    def reset_idx(self, env_ids: EnvIds) -> None:
        """Reset specified environments to initial states.
        
        Performs complete reset of specified environments including:
            - Curriculum updates (terrain and command curricula if enabled)
            - Command resampling
            - DOF state reset
            - Root state (base position/velocity) reset
            - Buffer resets (actions, episode length, failure counter)
            - Episode statistics logging
        
        Args:
            env_ids: Integer tensor of shape (num_reset_envs,) containing indices
                of environments to reset. Can be empty tensor.
        
        Example:
            >>> env_ids = torch.tensor([0, 5, 10], device="cuda")
            >>> env.reset_idx(env_ids)
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length ==0):
            self._update_command_curriculum(env_ids)

        self._resample_commands(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.simulator.reset_idx(env_ids)

        # reset buffers
        self.llast_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.fail_buf[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.simulator.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self) -> None:
        """Compute rewards for all environments.
        
        Iterates through all reward functions with non-zero scales (defined in config),
        computes each reward term, scales it by dt, and accumulates into the total reward.
        Optionally clips rewards to positive values and adds termination penalty.
        
        The reward structure is defined by cfg.rewards.scales in the configuration.
        Each reward function must be implemented as a method named _reward_<name>.
        
        Updates:
            - rew_buf: Total reward of shape (num_envs,).
            - episode_sums: Dictionary tracking cumulative rewards per term.
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination(
            ) * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self) -> None:
        """Compute observations for all environments.
        
        Constructs observation tensors from simulator states and applies normalization
        scales. The default observation includes:
            - Base linear velocity (3): scaled by obs_scales.lin_vel
            - Projected gravity (3): body z-axis in world frame
            - Base angular velocity (3): scaled by obs_scales.ang_vel
            - Commands (3): linear x, linear y, angular yaw velocities
            - DOF positions (num_dofs): deviation from default, scaled
            - DOF velocities (num_dofs): scaled by obs_scales.dof_vel
            - Actions (num_actions): previous action for history
        
        Optionally adds:
            - Height measurements: if cfg.terrain.measure_heights is True
            - Observation noise: if cfg.noise.add_noise is True
            - Privileged observations: if using asymmetric actor-critic
        
        Updates:
            - obs_buf: Observation buffer of shape (num_envs, obs_dim).
            - privileged_obs_buf: Privileged observations of shape (num_envs, privileged_obs_dim)
                or None if not using asymmetric actor-critic.
        """
        self.obs_buf = torch.cat((self.simulator.base_lin_vel * self.obs_scales.lin_vel,                    # 3
                                    self.simulator.projected_gravity,                                         # 3
                                    self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
                                    self.commands[:, :3] * self.commands_scale,                   # 3
                                    (self.simulator.dof_pos - self.simulator.default_dof_pos) 
                                      * self.obs_scales.dof_pos, # num_dofs
                                    self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
                                    self.actions                                                    # num_actions
                                    ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # === OBSERVATION BUFFER VALIDATION ===
        # Ensure observation buffer matches expected size
        assert self.obs_buf.shape[1] == self.num_obs, (
            f"Observation buffer size mismatch: expected {self.num_obs}, got {self.obs_buf.shape[1]}. "
            f"Check compute_observations() implementation matches cfg.env.num_observations. "
            f"Observation components: base_lin_vel(3) + gravity(3) + base_ang_vel(3) + commands(3) + "
            f"dof_pos({self.num_actions}) + dof_vel({self.num_actions}) + actions({self.num_actions})"
        )

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - \
                             1) * self.noise_scale_vec

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                (
                    self.simulator.base_lin_vel * self.obs_scales.lin_vel,
                    self.simulator.base_ang_vel * self.obs_scales.ang_vel,
                    self.simulator.projected_gravity,
                    self.commands[:, :3] * self.commands_scale,
                    (self.simulator.dof_pos - self.simulator.default_dof_pos) * \
                     self.obs_scales.dof_pos,
                    self.simulator.dof_vel * self.obs_scales.dof_vel,
                    self.actions,
                    self.last_actions,
                    self.simulator.dr_friction_values,        # 1
                    self.simulator.dr_added_base_mass,        # 1
                    self.simulator.dr_base_com_bias,          # 3
                    self.simulator.dr_rand_push_vels[:, :2],  # 2
                ),
                dim=-1,
            )
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                    1) - 0.5 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

    def set_viewer_camera(self, pos: np.ndarray, lookat: np.ndarray) -> None:
        """Set viewer camera position and orientation.
        
        Args:
            pos: Camera position in world frame as numpy array of shape (3,).
            lookat: Point to look at in world frame as numpy array of shape (3,).
        
        Example:
            >>> pos = np.array([2.0, 0.0, 1.0])
            >>> lookat = np.array([0.0, 0.0, 0.5])
            >>> env.set_viewer_camera(pos, lookat)
        """
        self.simulator.set_viewer_camera(eye=pos, target=lookat)

    # ------------- Callbacks (Protected Function) --------------
    
    def _pre_sim_step(self, actions: Action) -> Action:
        """ Callback called at the beginning of the step function, before stepping the simulation
        """
        clip_actions = self.cfg.normalization.clip_actions
        actions = torch.clip(
            actions, -clip_actions, clip_actions).to(self.device)
        # update history of actions
        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.actions[:] = actions[:]
        # during training, the camera follows the first environment
        if not self.debug and not self.headless:
            pos = self.simulator.base_pos[0].cpu().numpy() + np.array(self.cfg.viewer.pos)
            lookat = self.simulator.base_pos[0].cpu().numpy() + np.array(self.cfg.viewer.lookat)
            self.set_viewer_camera(pos, lookat)
        
        return actions
    
    def _update_terrain_curriculum(self, env_ids: EnvIds) -> None:
        """ Implements the game-inspired curriculum.

        Args:
            env_ids: ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(
            self.simulator.base_pos[env_ids, :2] - self.simulator.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.simulator._terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(
            self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        
        self.simulator.update_terrain_curriculum(env_ids, move_up, move_down)
    
    def _reset_dofs(self, env_ids: EnvIds) -> None:
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_pos[:, :] = self.simulator.default_dof_pos[:] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_actions), self.device)
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _reset_root_states(self, env_ids: EnvIds) -> None:
        # base pos
        if self.simulator.custom_origins:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
            base_pos[:, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
        # base quat
        roll_scale = self.cfg.init_state.roll_random_scale
        pitch_scale = self.cfg.init_state.pitch_random_scale
        yaw_scale = self.cfg.init_state.yaw_random_scale
        base_quat = quat_from_euler_xyz(
            torch_rand_float(-roll_scale, roll_scale, (len(env_ids), 1), self.device).squeeze(1),
            torch_rand_float(-pitch_scale, pitch_scale, (len(env_ids), 1), self.device).squeeze(1),
            torch_rand_float(-yaw_scale, yaw_scale, (len(env_ids), 1), self.device).squeeze(1)
        )
        # base lin vel
        base_lin_vel = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        # base ang vel
        base_ang_vel = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        self.simulator.reset_root_states(env_ids, 
                                         base_pos, 
                                         base_quat, 
                                         base_lin_vel, 
                                         base_ang_vel)

    def _post_physics_step_callback(self) -> None:
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.simulator.base_quat, self.forward_vec)
            self.heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - self.heading), self.cfg.commands.ranges.ang_vel_yaw[0], 
                                                                 self.cfg.commands.ranges.ang_vel_yaw[1])

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self.simulator.push_robots()
            # print(f"pushing robots")
        if self.cfg.domain_rand.push_links and (self.common_step_counter % self.cfg.domain_rand.push_links_interval == 0):
            self.simulator.push_links()
            # print(f"pushing links")
        
    def _resample_commands(self, env_ids: EnvIds) -> None:
        """ Randommly select commands of some environments

        Args:
            env_ids: Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids),1), self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids),1), self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        if np.random.rand() < self.cfg.commands.zero_cmd_prob:
            self.commands[env_ids, :3] *= 0.0  # set command to zero with some probability, to encourage the robot to learn to stand still
        
        # set small commands to zero
        self.commands[env_ids, :3] *= (torch.norm(
            self.commands[env_ids, :3], dim=1) > 0.2).unsqueeze(1)

    def _update_command_curriculum(self, env_ids: EnvIds) -> None:
        """ Implements a curriculum of increasing commands

        Args:
            env_ids: ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > \
                self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def _get_noise_scale_vec(self) -> Tensor:
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Returns:
            Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * \
            noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:24] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    # ----------------------------------------
    def _init_buffers(self) -> None:
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec()
        self.forward_vec = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.forward_vec[:, 0] = 1.0
        self.fail_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        
        # === COMMAND BUFFER VALIDATION ===
        assert hasattr(self.cfg.commands, 'num_commands'), (
            f"Config.commands missing 'num_commands'. "
            f"Required for command buffer allocation. "
            f"Add: cfg.commands.num_commands = 4"
        )
        self.commands = torch.zeros(
            (self.num_envs, self.cfg.commands.num_commands), device=self.device, dtype=torch.float)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel],
                                           device=self.device, dtype=torch.float,
                                           requires_grad=False)
        
        # === ACTION BUFFER VALIDATION ===
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros_like(self.actions)
        self.llast_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)  # last last actions
        
        # === FEET INDICES VALIDATION ===
        assert hasattr(self.simulator, 'feet_indices'), (
            "Simulator missing 'feet_indices' attribute. "
            "This attribute should be set during simulator initialization to identify foot bodies. "
            "Check your simulator implementation."
        )
        assert len(self.simulator.feet_indices) > 0, (
            f"Simulator.feet_indices is empty. "
            f"Expected at least one foot index for contact detection. "
            f"Check robot URDF configuration and simulator setup."
        )
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.simulator.feet_indices)), device=self.device, dtype=torch.float)
        self.last_contacts = torch.zeros((self.num_envs, len(self.simulator.feet_indices)), device=self.device, dtype=torch.int)

    def _prepare_reward_function(self) -> None:
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # === REWARD SCALES VALIDATION ===
        assert len(self.reward_scales) > 0, (
            "No reward scales defined in config. "
            "Add at least one reward scale in cfg.rewards.scales, e.g.: "
            "cfg.rewards.scales.tracking_lin_vel = 1.0"
        )
        
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale ==0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name =="termination":
                continue
            self.reward_names.append(name)
            method_name = '_reward_' + name
            
            # === REWARD FUNCTION VALIDATION ===
            assert hasattr(self, method_name), (
                f"Reward function '{method_name}' not found for reward scale '{name}'. "
                f"You must implement a method '_reward_{name}()' that returns a tensor of shape (num_envs,). "
                f"Either implement the method or remove '{name}' from cfg.rewards.scales."
            )
            self.reward_functions.append(getattr(self, method_name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _parse_cfg(self, cfg: LeggedRobotCfg) -> None:
        # === SIMULATION TIMING VALIDATION ===
        assert hasattr(cfg, 'sim'), (
            "Config missing 'sim' section. "
            "Required for simulation timing. "
            "Add: cfg.sim.dt = 0.005"
        )
        assert hasattr(cfg.sim, 'dt'), (
            "Config.sim missing 'dt'. "
            "Required for simulation timestep. "
            "Add: cfg.sim.dt = 0.005"
        )
        assert hasattr(cfg, 'control'), (
            "Config missing 'control' section. "
            "Required for control timing. "
            "Add: cfg.control.decimation = 4"
        )
        assert hasattr(cfg.control, 'decimation'), (
            "Config.control missing 'decimation'. "
            "Required for control frequency. "
            "Add: cfg.control.decimation = 4"
        )
        assert cfg.sim.dt > 0, (
            f"Config.sim.dt must be positive, got {cfg.sim.dt}. "
            f"Typical values: 0.005 (5ms) to 0.02 (20ms)."
        )
        assert cfg.control.decimation >= 1, (
            f"Config.control.decimation must be >= 1, got {cfg.control.decimation}. "
            f"This determines how many simulation steps per control step."
        )
        
        self.dt = self.cfg.sim.dt * self.cfg.control.decimation
        self.debug = self.cfg.env.debug
        # use self-implemented pd controller
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.push_links_interval = np.ceil(self.cfg.domain_rand.push_links_interval_s / self.dt)
        
    # ------------ reward functions----------------
    def _reward_lin_vel_z(self) -> Reward:
        # Penalize z axis base linear velocity
        return torch.square(self.simulator.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self) -> Reward:
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.simulator.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self) -> Reward:
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.simulator.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self) -> Reward:
        # Penalize base height away from target
        base_height = torch.mean(self.simulator.base_pos[:, 2].unsqueeze(
            1) - self.simulator.measured_heights, dim=1)
        # print(f"base height: {base_height}")
        rew = torch.square(base_height - self.cfg.rewards.base_height_target)
        return rew

    def _reward_torques(self) -> Reward:
        # Penalize torques
        return torch.sum(torch.square(self.simulator.torques), dim=1)

    def _reward_dof_vel(self) -> Reward:
        # Penalize dof velocities
        return torch.sum(torch.square(self.simulator.dof_vel), dim=1)
    
    def _reward_dof_power(self) -> Reward:
        # Penalize power consumption
        return torch.sum(torch.abs(self.simulator.torques * self.simulator.dof_vel), dim=1)

    def _reward_dof_acc(self) -> Reward:
        # Penalize dof accelerations
        return torch.sum(torch.square((self.simulator.last_dof_vel - 
                                       self.simulator.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self) -> Reward:
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self) -> Reward:
        # Penalize action smoothness
        action_smoothness_cost = torch.sum(torch.square(
            self.actions - 2*self.last_actions + self.llast_actions), dim=-1)
        return action_smoothness_cost

    def _reward_collision(self) -> Reward:
        # Penalize collisions on selected bodies
        # print(f"contacts: {(torch.norm(self.simulator.link_contact_forces[0, self.simulator.penalized_contact_indices, :], dim=-1) > 0.1)}")
        rew = torch.sum(1.*(self.penalized_bodies_force_norm > 10.0), dim=1)
        # print(f"collision reward: {rew[0]}")
        return rew

    def _reward_termination(self) -> Reward:
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self) -> Reward:
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_vel_limits(self) -> Reward:
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(self.simulator.torques) - self.simulator.torques_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self) -> Reward:
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.simulator.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self) -> Reward:
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.simulator.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self) -> Reward:
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.simulator.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self) -> Reward:
        # Reward long steps
        contact = self.feet_max_force_z > 10.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.2  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_dof_vel_stand_still(self) -> Reward:
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.simulator.dof_vel), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.2)

    def _reward_dof_pos_stand_still(self) -> Reward:
        # Penalize position deviation at zero commands
        return torch.sum(torch.square(self.simulator.dof_pos - self.simulator.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.2)
    
    def _reward_feet_contact_stand_still(self) -> Reward:
        # Encourage feet contact with the ground at zero commands
        contacts = self.feet_max_force_z > 10.0
        full_contact = torch.sum(1.*contacts, dim=1)==len(self.simulator.feet_contact_indices)
        return 1.0 * full_contact * (torch.norm(self.commands[:, :3], dim=1) < 0.2)
    
    def _reward_dof_close_to_default(self) -> Reward:
        # Penalize dof position deviation from default
        return torch.sum(torch.square(self.simulator.dof_pos - self.simulator.default_dof_pos), dim=1)
    
    def _reward_dof_close_to_default_stand_still(self) -> Reward:
        # Penalize dof position deviation from default at zero commands
        return torch.sum(torch.square(self.simulator.dof_pos - self.simulator.default_dof_pos), dim=1) \
                * (torch.norm(self.commands[:, :3], dim=1) < 0.2)

    def _reward_foot_clearance(self) -> Reward:
        # Encourage feet to be close to desired height while swinging
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        # print(f"feet pos: {self.simulator.feet_pos[:, :, 2]}")
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.simulator.feet_pos[:, :, 2] -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_foot_landing_vel(self) -> Reward:
        z_vels = self.simulator.feet_vel[:, :, 2]
        contacts = self.feet_max_force_z > 10.0
        about_to_land = ((self.simulator.feet_pos[:, :, 2] -
                          self.cfg.rewards.foot_height_offset) <
                         self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(
            about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward
    
    def _reward_keep_balance(self) -> Reward:
        return torch.ones(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
    
    def _reward_foot_acc(self) -> Reward:
        # reward for foot acceleration
        foot_acc = (self.simulator.feet_vel - self.simulator.last_feet_vel) / self.dt
        return torch.sum(torch.square(foot_acc), dim=(1, 2))
    
    def _reward_feet_slip(self) -> Reward:
        # penalize foot slip when in contact with the ground
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        contacts = self.feet_max_force_z > 10.0
        slip_penalty = torch.sum(foot_vel_xy_norm * contacts, dim=1)
        return slip_penalty
    
    def _reward_no_fly(self) -> Reward:
        # Encourage only one foot in the air at a time
        contacts = self.feet_max_force_z > 10.0
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        # if the dim of link_contact_forces is 4. It has history, shape is (num_envs, history_length, num_links, 3)
        if len(self.simulator.link_contact_forces.shape) == 4:
            feet_max_norm_xy = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.feet_contact_indices, :2], dim=-1), dim=1)[0]
            feet_max_force_z = torch.max(torch.abs(self.simulator.link_contact_forces[:, :, self.simulator.feet_contact_indices, 2]), dim=1)[0]
        else:
            feet_max_norm_xy = torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, :2], dim=-1)
            feet_max_force_z = torch.abs(self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2])

        rew = torch.any(feet_max_norm_xy > 4 *feet_max_force_z, dim=1)
        return rew.float()
