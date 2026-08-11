from legged_gym.envs.base.legged_robot_amp import LeggedRobotAMP
import torch
from legged_gym.utils.math_utils import quat_rotate_inverse, torch_rand_float
from collections import deque
import numpy as np

class K1AMP(LeggedRobotAMP):
    
    def compute_observations(self):
        
        key_body_pos_relative_to_base = self.simulator.key_body_pos - \
                self.simulator.base_pos.unsqueeze(1)
        self._calc_periodic_reward_obs()
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.simulator.projected_gravity,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            (self.simulator.dof_pos - self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        
        domain_randomization_info = torch.cat((
            self.simulator.dr_friction_values,        # 1
            self.simulator.dr_added_base_mass,        # 1
            self.simulator.dr_rand_push_vels[:, :2],  # 2
            self.simulator.dr_base_com_bias,          # 3
            self.simulator.dr_kp_scale,               # num_dofs
            self.simulator.dr_kd_scale,               # num_dofs
        ), dim=-1)
        
        single_critic_obs = torch.cat((
            obs_buf,                                               # num_obs
            self.simulator.base_lin_vel * self.obs_scales.lin_vel, # 3
            key_body_pos_relative_to_base.flatten(start_dim=1),    # num_key_bodies * 3
            domain_randomization_info,                             # 51
            self.simulator.feet_pos[:, :, 2],                      # 2
            self.clock_input,                                      # 4
        ), dim=-1)
        
        self.critic_obs_deque.append(single_critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec
        
        # push obs_buf to obs_history
        self.obs_history_deque.append(obs_buf)
        self.obs_buf = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
        
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
                    self.cfg.env.num_single_critic_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        # Periodic Reward Framework
        self.theta = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.theta[:, 0] = self.cfg.rewards.periodic_reward_framework.theta_left
        self.theta[:, 1] = self.cfg.rewards.periodic_reward_framework.theta_right
        self.gait_time = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.phi = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.gait_period = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
        self.gait_period[:] = self.cfg.rewards.periodic_reward_framework.gait_period
        self.clock_input = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
        )
        self.b_swing = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.b_swing[:] = self.cfg.rewards.periodic_reward_framework.b_swing * 2 * torch.pi
    
    
    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > \
                self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    
    def _reset_dofs(self, env_ids):
        dof_pos = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float,
                              device=self.device, requires_grad=False)
        dof_vel = torch.zeros((len(env_ids), self.num_actions), dtype=torch.float,
                              device=self.device, requires_grad=False)
        default = self.simulator.default_dof_pos
        # Shoulder_Pitch(2,6), Shoulder_Roll(3,7), Elbow_Yaw(5,9)
        dof_pos[:, [2,3,5,6,7,9]] = default[:, [2,3,5,6,7,9]] + torch_rand_float(-0.1, 0.1, (len(env_ids), 6), self.device)
        # Hip_Pitch(10,16), Hip_Roll(11,17)
        dof_pos[:, [10,11,16,17]] = default[:, [10,11,16,17]] + torch_rand_float(-0.1, 0.1, (len(env_ids), 4), self.device)
        # Knee_Pitch(13,19)
        dof_pos[:, [13,19]] = default[:, [13,19]] + torch_rand_float(-0.1, 0.3, (len(env_ids), 2), self.device)
        # Ankle_Pitch(14,20)
        dof_pos[:, [14,20]] = default[:, [14,20]] + torch_rand_float(-0.1, 0.1, (len(env_ids), 2), self.device)
        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _reset_dofs_from_reference_motion(self, env_ids, ref_motions=None):
        """Reset the dof positions and velocities of the robots in env_ids to the reference motion at random time steps

        Args:
            env_ids (torch.Tensor): Tensor of shape (num_envs_to_reset,) containing the ids of the envs to reset
        """
        ref_dof_pos = self.amp_loader.get_dof_pos_batch(ref_motions)
        ref_dof_vel = self.amp_loader.get_dof_vel_batch(ref_motions)
        self.simulator.reset_dofs(env_ids, ref_dof_pos, ref_dof_vel)
        
    def _reset_root_states_from_reference_motion(self, env_ids, ref_motions=None):
        """Reset the root positions, orientations, linear and angular velocities of the robots in env_ids to the reference motion at random time steps

        Args:
            env_ids (torch.Tensor): Tensor of shape (num_envs_to_reset,) containing the ids of the envs to reset
        """
        ref_base_pos = self.amp_loader.get_base_pos_batch(ref_motions)
        ref_base_pos[:, 2] = self.simulator.base_init_pos[2]
        base_pos = ref_base_pos + self.simulator.env_origins[env_ids]
        ref_base_rot = self.amp_loader.get_base_rot_batch(ref_motions)
        ref_base_lin_vel = self.amp_loader.get_base_lin_vel_batch(ref_motions)
        ref_base_ang_vel = self.amp_loader.get_base_ang_vel_batch(ref_motions)
        self.simulator.reset_root_states(env_ids, base_pos, ref_base_rot, ref_base_lin_vel, ref_base_ang_vel)
    
    def reset_idx(self, env_ids):
        # Periodic Reward Framework phi cycle
        # step after computing reward but before resetting the env
        self.gait_time += self.dt
        # +self.dt/2 in case of float precision errors
        is_over_limit = (self.gait_time >= (self.gait_period - self.dt / 2))
        over_limit_indices = is_over_limit.nonzero(as_tuple=False).flatten()
        self.gait_time[over_limit_indices] = 0.0
        self.phi = self.gait_time / self.gait_period
        
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        # Periodic Reward Framework buffer reset
        self.theta[env_ids, 0] = self.cfg.rewards.periodic_reward_framework.theta_left + \
            torch.rand(len(env_ids), 1, device=self.device).squeeze(1)  # add small random offset
        self.theta[env_ids, 1] = self.theta[env_ids, 0] + (self.cfg.rewards.periodic_reward_framework.theta_right -
                                                           self.cfg.rewards.periodic_reward_framework.theta_left)
        self.gait_time[env_ids] = torch.rand(len(env_ids), 1, device=self.device) * self.gait_period[env_ids]
        self.phi[env_ids] = self.gait_time[env_ids] / self.gait_period[env_ids]
        self.clock_input[env_ids, :] = 0.0
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        # Periodic Reward Framework. Constants are init here.
        self.kappa = self.cfg.rewards.periodic_reward_framework.kappa
        self.gait_function_type = self.cfg.rewards.periodic_reward_framework.gait_function_type
        self.a_swing = 0.0
        self.b_stance = 2 * torch.pi # a_stance(start of stance) = b_swing
    
    def _calc_periodic_reward_obs(self):
        """Calculate the periodic reward observations.
        """
        for i in range(2):
            self.clock_input[:, i] = torch.sin(2 * torch.pi * (self.phi + self.theta[:, i].unsqueeze(1))).squeeze(-1)
            self.clock_input[:, i + 2] = torch.cos(2 * torch.pi * (self.phi + self.theta[:, i].unsqueeze(1))).squeeze(-1)
    
    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9 + self.num_actions:9 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9 + 2 * self.num_actions:9 + 3 * self.num_actions] = 0.  # previous actions
        return noise_vec
    
    def _uniped_periodic_gait(self, foot_type):
        # q_frc and q_spd
        if foot_type == "left":
            q_frc = self.feet_force_norm[:, 0].view(-1, 1)
            q_spd = torch.norm(
                self.simulator.feet_vel[:, 0, :], dim=-1).view(-1, 1) # sequence of feet_pos is FL, FR, RL, RR
            # size: num_envs; need to reshape to (num_envs, 1), or there will be error due to broadcasting
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 0].unsqueeze(1)) % 1.0
        elif foot_type == "right":
            q_frc = self.feet_force_norm[:, 1].view(-1, 1)
            q_spd = torch.norm(
                self.simulator.feet_vel[:, 1, :], dim=-1).view(-1, 1)
            # modulo phi over 1.0 to get cicular phi in [0, 1.0]
            phi = (self.phi + self.theta[:, 1].unsqueeze(1)) % 1.0
        
        phi *= 2 * torch.pi  # convert phi to radians
        
        if self.gait_function_type == "smooth":
            # coefficient
            c_swing_spd = 0  # speed is not penalized during swing phase
            c_swing_frc = -1  # force is penalized during swing phase
            c_stance_spd = -1  # speed is penalized during stance phase
            c_stance_frc = 0  # force is not penalized during stance phase
            
            # clip the value of phi to [0, 1.0]. The vonmises function in scipy may return cdf outside [0, 1.0]
            F_A_swing = torch.clip(torch.tensor(vonmises.cdf(loc=self.a_swing, 
                kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
            F_B_swing = torch.clip(torch.tensor(vonmises.cdf(loc=self.b_swing.cpu(), 
                kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)
            F_A_stance = F_B_swing
            F_B_stance = torch.clip(torch.tensor(vonmises.cdf(loc=self.b_stance,
                kappa=self.kappa, x=phi.cpu()), device=self.device), 0.0, 1.0)

            # calc the expected C_spd and C_frc according to the formula in the paper
            exp_swing_ind = F_A_swing * (1 - F_B_swing)
            exp_stance_ind = F_A_stance * (1 - F_B_stance)
            exp_C_spd_ori = c_swing_spd * exp_swing_ind + c_stance_spd * exp_stance_ind
            exp_C_frc_ori = c_swing_frc * exp_swing_ind + c_stance_frc * exp_stance_ind

            # just the code above can't result in the same reward curve as the paper
            # a little trick is implemented to make the reward curve same as the paper
            # first let all envs get the same exp_C_frc and exp_C_spd
            exp_C_frc = -0.5 + (-0.5 - exp_C_spd_ori)
            exp_C_spd = exp_C_spd_ori
            # select the envs that are in swing phase
            is_in_swing = (phi >= self.a_swing) & (phi < self.b_swing)
            indices_in_swing = is_in_swing.nonzero(as_tuple=False).flatten()
            # update the exp_C_frc and exp_C_spd of the envs in swing phase
            exp_C_frc[indices_in_swing] = exp_C_frc_ori[indices_in_swing]
            exp_C_spd[indices_in_swing] = -0.5 + \
                (-0.5 - exp_C_frc_ori[indices_in_swing])

            # Judge if it's the standing gait
            is_standing = (self.b_swing[:] == self.a_swing).nonzero(
                as_tuple=False).flatten()
            exp_C_frc[is_standing] = 0
            exp_C_spd[is_standing] = -1
        elif self.gait_function_type == "step":
            ''' ***** Step Gait Indicator ***** '''
            exp_C_frc = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
            exp_C_spd = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device)
            
            swing_indices = (phi >= self.a_swing) & (phi < self.b_swing)
            swing_indices = swing_indices.nonzero(as_tuple=False).flatten()
            stance_indices = (phi >= self.b_swing) & (phi < self.b_stance)
            stance_indices = stance_indices.nonzero(as_tuple=False).flatten()
            exp_C_frc[swing_indices, :] = -1
            exp_C_spd[swing_indices, :] = 0
            exp_C_frc[stance_indices, :] = 0
            exp_C_spd[stance_indices, :] = -1

        return exp_C_spd * q_spd + exp_C_frc * q_frc, \
            exp_C_spd.type(dtype=torch.float), exp_C_frc.type(dtype=torch.float)
    
    def _reward_biped_periodic_gait(self):
        biped_reward_left, self.exp_C_spd_left, self.exp_C_frc_left = self._uniped_periodic_gait(
            "left")
        biped_reward_right, self.exp_C_spd_right, self.exp_C_frc_right = self._uniped_periodic_gait(
            "right")
        # reward for the whole body
        biped_reward = biped_reward_left.flatten() + biped_reward_right.flatten()
        return torch.exp(biped_reward) * (torch.norm(self.commands[:, :3], dim=-1) > 0.2)  # only give reward when there is enough command to follow, otherwise the robot may learn to be static and get reward for free
    
    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.feet_force_norm > 10.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.4) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.2  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_feet_distance(self):
        '''reward for feet distance'''
        feet_xy_distance = torch.norm(
            self.simulator.feet_pos[:, 0, [0, 1]] - self.simulator.feet_pos[:, 1, [0, 1]], dim=-1)
        return torch.max(torch.zeros_like(feet_xy_distance),
                         self.cfg.rewards.foot_distance_threshold - feet_xy_distance)
    
    def _reward_foot_flat(self):
        """Encourage foot to be flat when contact with the ground
        """
        foot_contact = self.feet_force_norm > 10.0
        foot_quat = self.simulator.feet_quat
        # calculate world z axis in foot frame
        z_axis_world = torch.tensor([0., 0., 1.], device=self.device).repeat(foot_quat.shape[0], foot_quat.shape[1], 1)
        foot_z_axis = quat_rotate_inverse(foot_quat, z_axis_world)
        foot_tilt = torch.abs(foot_z_axis[:, :, 0]) + torch.abs(foot_z_axis[:, :, 1])  # x and y components
        rew_foot_flat = torch.exp(-foot_tilt / 0.1)
        return torch.sum(rew_foot_flat * foot_contact, dim=1)
    
    def _reward_hip_yaw_roll_pos(self):
        """Encourage hip yaw to be close to default position
        """
        hip_yaw_roll = self.simulator.dof_pos[:, [11,12,17,18]]
        return torch.sum(torch.square(hip_yaw_roll - self.simulator.default_dof_pos[:, [11,12,17,18]]), dim=1)
    
    def _reward_arm_pos(self):
        """Encourage arm joints to be close to default position
        """
        arm_joints = self.simulator.dof_pos[:, [2,3,4,5,6,7,8,9]]
        return torch.sum(torch.square(arm_joints - self.simulator.default_dof_pos[:, [2,3,4,5,6,7,8,9]]), dim=1)
    
    def _reward_head_pos(self):
        """Encourage head joints to be close to default position
        """
        head_joints = self.simulator.dof_pos[:, :2]
        return torch.sum(torch.square(head_joints - self.simulator.default_dof_pos[:, :2]), dim=1)
    
    def _reward_feet_slip(self):
        '''penalize foot slip when in contact with the ground'''
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        contacts = self.feet_max_force_z > 10.0
        slip_penalty = torch.sum(foot_vel_xy_norm * contacts, dim=1)
        return slip_penalty
    
    def _reward_foot_landing_vel(self):
        z_vels = self.simulator.feet_vel[:, :, 2]
        contacts = self.feet_max_force_z > 10.0
        about_to_land = ((self.simulator.feet_pos[:, :, 2] -
                          self.cfg.rewards.foot_height_offset) <
                         self.cfg.rewards.about_landing_threshold) & (~contacts) & (z_vels < 0.0)
        landing_z_vels = torch.where(
            about_to_land, z_vels, torch.zeros_like(z_vels))
        reward = torch.sum(torch.square(landing_z_vels), dim=1)
        return reward