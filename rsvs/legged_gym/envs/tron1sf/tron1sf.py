from legged_gym import *
from time import time
import numpy as np

import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import *
from legged_gym.utils.helpers import class_to_dict
from collections import deque

class TRON1SF(LeggedRobot):
    
    def compute_observations(self):
        obs_buf = torch.cat((
                                self.commands[:, :3] * self.commands_scale,                   # 3
                                self.simulator.projected_gravity,                             # 3
                                self.simulator.base_ang_vel * self.obs_scales.ang_vel,        # 3
                                (self.simulator.dof_pos - self.simulator.default_dof_pos) 
                                    * self.obs_scales.dof_pos,                                # num_dofs
                                self.simulator.dof_vel * self.obs_scales.dof_vel,             # num_dofs
                                self.actions                                                  # num_actions
                                ), dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            obs_buf = torch.cat((obs_buf, heights), dim=-1)

        if self.num_privileged_obs is not None:
            privileged_obs_buf = torch.cat(
                (
                    self.simulator.base_lin_vel * self.obs_scales.lin_vel, # 3
                    obs_buf,
                    self.last_actions,                      # num_actions
                    self.simulator.dr_friction_values,            # 1
                    self.simulator.dr_added_base_mass,        # 1
                    self.simulator.dr_base_com_bias,          # 3
                    self.simulator.dr_rand_push_vels[:, :2],  # 2
                    self.feet_air_time,                     # 2
                    self.simulator.dr_kp_scale,                 # num_actions
                    self.simulator.dr_kd_scale,                 # num_actions
                    self.simulator.dr_joint_armature,
                    self.simulator.dr_joint_friction,
                    self.simulator.dr_joint_damping,
                ),
                dim=-1,
            )
            # add perceptive inputs if not blind
            if self.cfg.terrain.measure_heights:
                heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                    1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
                privileged_obs_buf = torch.cat((privileged_obs_buf, heights), dim=-1)
            
            self.critic_obs_deque.append(privileged_obs_buf)
            self.privileged_obs_buf = torch.cat(
                [self.critic_obs_deque[i]
                    for i in range(self.critic_obs_deque.maxlen)],
                dim=-1,
            )
                
        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - \
                             1) * self.noise_scale_vec
        
        self.obs_history_deque.append(obs_buf)
        # construct stacked observations
        self.obs_buf = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
        
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
    
    def _init_buffers(self):
        super()._init_buffers()
        # obs_history
        self.obs_history_deque = deque(maxlen=self.cfg.env.frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        # critic observation buffer
        self.critic_obs_deque = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_obs_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_privileged_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        # buffers for sit pose params
        self.sit_pos = torch.tensor(self.cfg.init_state.sit_pos, dtype=torch.float, device=self.device, requires_grad=False)
        self.sit_joint_angles = torch.tensor(
            [self.cfg.init_state.sit_joint_angles[name]
                for name in self.cfg.asset.dof_names],
            device=self.device,
            dtype=torch.float,
        )
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self._update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
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
        _ = np.random.random() # initialize the env at sit pose randomly
        if _ < self.cfg.init_state.sit_init_percent:
            self._reset_dofs_sit_pose(env_ids)
            self._reset_root_states_sit_pose(env_ids)
        else:
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

        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
        
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
        dof_pos[:, [0, 3]] = self.simulator.default_dof_pos[:, [0, 3]] + \
            torch_rand_float(-0.05, 0.05, (len(env_ids), 2), self.device) # abad
        dof_pos[:, [1, 4]] = self.simulator.default_dof_pos[:, [1, 4]] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), 2), self.device)   # hip
        dof_pos[:, [2, 5]] = self.simulator.default_dof_pos[:, [2, 5]] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), 2), self.device)   # knee
        dof_pos[:, [3, 6]] = self.simulator.default_dof_pos[:, [3, 6]] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), 2), self.device)   # ankle
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _reset_dofs_sit_pose(self, env_ids):
        dof_pos = self.sit_joint_angles.unsqueeze(0).repeat(len(env_ids), 1)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float, 
                              device=self.device, requires_grad=False)
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _reset_root_states_sit_pose(self, env_ids):
        if self.simulator.custom_origins:
            base_pos = self.sit_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
            base_pos[:, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            base_pos = self.sit_pos.reshape(1, -1).repeat(len(env_ids), 1)
            base_pos += self.simulator.env_origins[env_ids]
        # base quat
        base_quat = self.simulator.base_init_quat.reshape(1, -1).repeat(len(env_ids), 1)
        base_lin_vel = torch.zeros((len(env_ids), 3), dtype=torch.float, 
                                  device=self.device, requires_grad=False)
        base_ang_vel = torch.zeros((len(env_ids), 3), dtype=torch.float, 
                                  device=self.device, requires_grad=False)
        self.simulator.reset_root_states(env_ids, base_pos, base_quat, base_lin_vel, base_ang_vel)
    
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[9:17] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[17:25] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[25:33] = 0.  # previous actions
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[27:214] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec
    
    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.feet_force_norm > 10.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_feet_distance(self):
        '''reward for feet distance'''
        feet_xy_distance = torch.norm(
            self.simulator.feet_pos[:, 0, [0, 1]] - self.simulator.feet_pos[:, 1, [0, 1]], dim=-1)
        return torch.max(torch.zeros_like(feet_xy_distance),
                         self.cfg.rewards.foot_distance_threshold - feet_xy_distance)
        
    def _reward_no_fly(self):
        contacts = self.feet_max_force_z > 10.0
        single_contact = torch.sum(1.*contacts, dim=1)==1
        return 1.*single_contact
    
    def _reward_hip_pos_zero_command(self):
        """Penalize hip joint deviation from default position when no command is given
        """
        return torch.sum(torch.square(self.simulator.dof_pos[:, [1,5]] - 
                                      self.simulator.default_dof_pos[:, [1,5]]), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.1)
    
    def _reward_keep_ankle_pitch_zero_in_air(self):
        """Reward keeping ankle pitch close to zero when in the air
        """
        contacts = self.feet_max_force_z > 10.0
        ankle_pitch = torch.abs(self.simulator.dof_pos[:, 3]) * ~(contacts[:, 0]) + torch.abs(
            self.simulator.dof_pos[:, 7]) * ~contacts[:, 1]
        return torch.exp(-torch.abs(ankle_pitch) / 0.2)
    
    def _reward_foot_flat(self):
        """Encourage foot to be flat
        """
        foot_quat = self.simulator.feet_quat
        # calculate world z axis in foot frame
        z_axis_world = torch.tensor([0., 0., 1.], device=self.device).repeat(foot_quat.shape[0], foot_quat.shape[1], 1)
        foot_z_axis = quat_rotate_inverse(foot_quat, z_axis_world)
        foot_tilt = torch.abs(foot_z_axis[:, :, 0]) + torch.abs(foot_z_axis[:, :, 1])  # x and y components
        rew_foot_flat = torch.exp(-foot_tilt / 0.1)
        return torch.sum(rew_foot_flat, dim=1)