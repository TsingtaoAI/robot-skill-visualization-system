from legged_gym import SIMULATOR
import numpy as np
import os

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import torch_rand_float

class BipedalWalker(LeggedRobot):
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        # if the dim of link_contact_forces is 4, then it has history. shape [N, T, B, 3] (N: num_envs, T: history length, B: number of links with contact sensors)
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
        fail_buf |= self.simulator.projected_gravity[:, 2] > self.cfg.env.max_projected_gravity
        # hip saggital and transversal angle limits
        hip_saggital_indices = [1, 6] # 髋部侧摆自由度
        hip_transversal_indices = [2, 7] # 髋部内外旋自由度
        # 髋关节侧摆角度大于30度则终止
        hip_saggital_ang = torch.any(torch.abs(self.simulator.dof_pos[:, hip_saggital_indices]) > torch.pi/4, dim=1)
        hip_transversal_ang = torch.any(torch.abs(self.simulator.dof_pos[:, hip_transversal_indices]) > 0.15, dim=1)
        fail_buf |= hip_saggital_ang
        fail_buf |= hip_transversal_ang
        self.fail_buf += fail_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
        )
    
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        dof_pos[:, [0,5]] = self.simulator.default_dof_pos[:, [0,5]] + torch_rand_float(-0.2, 0.2, (len(env_ids), 2), device=self.device) # saggital
        dof_pos[:, [1,6]] = self.simulator.default_dof_pos[:, [1,6]] + torch_rand_float(-0.2, 0.2, (len(env_ids), 2), device=self.device) # frontal
        dof_pos[:, [2,7]] = self.simulator.default_dof_pos[:, [2,7]] + torch_rand_float(-0.05, 0.05, (len(env_ids), 2), device=self.device) # transversal
        dof_pos[:, [3,8]] = torch_rand_float(0.0, torch.pi/2, (len(env_ids), 2), device=self.device) # knee
        dof_pos[:, [4,9]] = torch_rand_float(-0.1, 0.1, (len(env_ids), 2), device=self.device) # ankle
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _reward_no_fly(self):
        contacts = self.feet_max_force_z > 10.0
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact