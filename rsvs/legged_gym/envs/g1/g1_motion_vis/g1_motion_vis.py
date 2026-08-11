from legged_gym import *
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot

class G1MotionVis(LeggedRobot):
    
    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        
        self.privileged_obs_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec