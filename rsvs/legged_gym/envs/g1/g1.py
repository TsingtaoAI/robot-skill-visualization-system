from legged_gym import *
import numpy as np
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import torch_rand_float
from .g1_config import G1RoughCfg


class G1Robot(LeggedRobot):
    """G1 humanoid using base LeggedRobot + IsaacGymSimulator (same stack as GO2)."""

    def _reset_dofs(self, env_ids):
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float,
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float,
                              device=self.device, requires_grad=False)
        default = self.simulator.default_dof_pos
        dof_pos[:, :] = default + torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_actions), self.device)
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)

    def _reset_root_states(self, env_ids):
        if self.simulator.custom_origins:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
            base_pos[:, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        else:
            base_pos = self.simulator.base_init_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
        base_quat = self.simulator.base_init_quat.reshape(1, -1).repeat(len(env_ids), 1)
        base_lin_vel = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        base_ang_vel = torch_rand_float(-0.5, 0.5, (len(env_ids), 3), self.device)
        self.simulator.reset_root_states(env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel)

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.  # commands
        noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9 + self.num_actions:9 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = 0.  # previous actions
        noise_vec[9 + 3 * self.num_actions:9 + 3 * self.num_actions + 2] = 0.  # sin/cos phase
        return noise_vec

    def _post_physics_step_callback(self):
        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        return super()._post_physics_step_callback()

    def compute_observations(self):
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        self.obs_buf = torch.cat((
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase
        ), dim=-1)
        self.privileged_obs_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase
        ), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        feet_idx = self.simulator.feet_contact_indices
        for i in range(len(feet_idx)):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.feet_max_force_z[:, i] > 10.0
            res += ~(contact ^ is_stance)
        return res

    def _reward_feet_swing_height(self):
        feet_idx = self.simulator.feet_contact_indices
        feet_pos = self.simulator.feet_pos
        contact = self.feet_force_norm > 10.0
        pos_error = torch.square(feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=1)

    def _reward_alive(self):
        return 1.0

    def _reward_contact_no_vel(self):
        feet_idx = self.simulator.feet_contact_indices
        feet_vel = self.simulator.feet_vel
        contact = self.feet_force_norm > 10.0
        contact_feet_vel = feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.simulator.dof_pos[:, [1, 2, 7, 8]]), dim=1)
