from legged_gym.envs.base.legged_robot import EnvIds
import numpy as np

from legged_gym import *
import torch
from legged_gym.envs.base.legged_robot_ts_depth import LeggedRobotTSDepth
from legged_gym.utils.math_utils import torch_rand_float

class Go2TSDepth(LeggedRobotTSDepth):
    
    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                               # 3
            self.simulator.projected_gravity,                                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                    # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,                                                  # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions * self.obs_scales.actions                                    # num_actions
        ), dim=-1)
        
        domain_randomization_info = torch.cat((
                    self.simulator.dr_friction_values,            # 1
                    self.simulator.dr_added_base_mass,            # 1
                    self.simulator.dr_base_com_bias,              # 3
                    self.simulator.dr_rand_push_vels[:, :2],      # 2
                    self.simulator.dr_kp_scale,                   # num_actions
                    self.simulator.dr_kd_scale                    # num_actions
            ), dim=-1)
        
        # Critic observation
        critic_obs = torch.cat((
            self.obs_buf,                 # num_observations
            domain_randomization_info,    # 31
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,     # 3
        ), dim=-1)
        if self.cfg.asset.obtain_link_contact_states:
            critic_obs = torch.cat(
                (
                    critic_obs,                          # previous
                    self.simulator.link_contact_states,  # contact states of thighs, calfs and feet (4+4+4)=12
                ),
                dim=-1,
            )
        if self.cfg.terrain.obtain_terrain_info_around_feet:
            height_relative_to_feet = (self.simulator.feet_pos[:, :, 2].unsqueeze(-1) - 
                                       self.simulator.height_around_feet).clip(-1, 1.).flatten(1, 2) # 9*number of feet=36
            critic_obs = torch.cat((critic_obs, 
                                    height_relative_to_feet,   # 36
                                    self.simulator.normal_vector_around_feet # 12
                                    ), dim=-1)
        if self.cfg.terrain.measure_heights: # 144
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.4 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            critic_obs = torch.cat((critic_obs, heights), dim=-1)
        self.critic_obs_deque.append(critic_obs)
        self.critic_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) -
                             1) * self.noise_scale_vec
        
        # Privileged observation, for privileged encoder
        self.privileged_obs_buf = torch.cat(
            (
                domain_randomization_info,                       # 32
                self.simulator.base_lin_vel * self.obs_scales.lin_vel,     # 3
            ),
            dim=-1,
        )
        if self.cfg.terrain.obtain_terrain_info_around_feet:
            self.privileged_obs_buf = torch.cat((
                                    self.privileged_obs_buf, 
                                    height_relative_to_feet,   # 36
                                    self.simulator.normal_vector_around_feet # 12
                                    ), dim=-1)
        if self.cfg.asset.obtain_link_contact_states:
            self.privileged_obs_buf = torch.cat(
                (
                    self.privileged_obs_buf,                   # previous
                    self.simulator.link_contact_states,        # contact states of thighs, calfs and feet (4+4+4)=12
                ),
                dim=-1,
            )
        if self.cfg.terrain.measure_heights: # 144
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
    
    def check_termination(self) -> None:
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
        
        if self.cfg.terrain.curriculum:
            # terminate if the robot's base height is below 0.0 on terrains with gaps, to prevent the robot from keep falling
            fail_buf[self.stepping_stones_env_ids] |= \
                self.simulator.base_pos[self.stepping_stones_env_ids, 2] < 0.0
            fail_buf[self.gaps_env_ids] |= \
                self.simulator.base_pos[self.gaps_env_ids, 2] < 0.0
            fail_buf[self.high_platform_gaps_env_ids] |= \
                self.simulator.base_pos[self.high_platform_gaps_env_ids, 2] < self.high_platform_gaps_termination_height[self.high_platform_gaps_env_ids]
        
        self.fail_buf += fail_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        # print(f"time out: {self.time_out_buf}")
        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
        )
    
    def _init_buffers(self):
        super()._init_buffers()
        # identify env ids for different terrain types
        if self.cfg.terrain.curriculum:
            terrain_types = self.simulator.terrain_types
            terrain_type_bounds = torch.cumsum(torch.tensor(self.cfg.terrain.terrain_proportions), dim=0) * self.cfg.terrain.num_cols
            self.slope_env_ids = ((terrain_types >=0) & (terrain_types < terrain_type_bounds[0])).nonzero(as_tuple=False).flatten()
            self.stairs_env_ids = ((terrain_types >= terrain_type_bounds[1]) & (terrain_types < terrain_type_bounds[3])).nonzero(as_tuple=False).flatten()
            self.discrete_env_ids = ((terrain_types >= terrain_type_bounds[3]) & (terrain_types < terrain_type_bounds[4])).nonzero(as_tuple=False).flatten()
            self.stepping_stones_env_ids = ((terrain_types >= terrain_type_bounds[4]) & (terrain_types < terrain_type_bounds[5])).nonzero(as_tuple=False).flatten()
            self.gaps_env_ids = ((terrain_types >= terrain_type_bounds[5]) & (terrain_types < terrain_type_bounds[6])).nonzero(as_tuple=False).flatten()
            self.pits_env_ids = ((terrain_types >= terrain_type_bounds[6]) & (terrain_types < terrain_type_bounds[7])).nonzero(as_tuple=False).flatten()
            self.high_platform_env_ids = ((terrain_types >= terrain_type_bounds[7]) & (terrain_types < terrain_type_bounds[8])).nonzero(as_tuple=False).flatten()
            self.high_platform_gaps_env_ids = ((terrain_types >= terrain_type_bounds[8]) & (terrain_types < terrain_type_bounds[9])).nonzero(as_tuple=False).flatten()
            # identify env ids for all heading command (others have fixed heading command specified in config)
            self.all_heading_env_ids = torch.cat((
                self.slope_env_ids,
                self.stairs_env_ids,
                self.discrete_env_ids,
                self.gaps_env_ids,
                self.pits_env_ids, # elementary terrains with all heading commands
            ))
            # identity termination base height for high_platform_gaps terrain
            difficulty = self.simulator.terrain_levels / self.cfg.terrain.num_rows
            self.high_platform_gaps_termination_height = eval(
                self.cfg.terrain.terrain_curriculum_difficulty["high_platform_gaps_params"]["high_platform_height"])
    
    def _resample_commands(self, env_ids: EnvIds) -> None:
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids),1), self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids),1), self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            if self.cfg.terrain.curriculum:
                # override heading command of some envs with all range heading command
                env_ids_for_heading = torch.tensor([env_id for env_id in env_ids if env_id in self.all_heading_env_ids], device=self.device)
                if len(env_ids_for_heading) > 0:
                    self.commands[env_ids_for_heading, 3] = torch_rand_float(-3.14, 3.14, (len(env_ids_for_heading), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        if np.random.rand() < self.cfg.commands.zero_cmd_prob:
            self.commands[env_ids, :3] *= 0.0  # set command to zero with some probability, to encourage the robot to learn to stand still
        
        # set small commands to zero
        self.commands[env_ids, :3] *= (torch.norm(
            self.commands[env_ids, :3], dim=1) > 0.2).unsqueeze(1)
    
    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > \
                self.cfg.commands.curriculum_threshold * self.reward_scales["tracking_lin_vel"]:
            # only increase upper bound of forward velocity command
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
    
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
        dof_pos[:, [0, 3, 6, 9]] = self.simulator.default_dof_pos[:, [0, 3, 6, 9]] + \
            torch_rand_float(-0.2, 0.2, (len(env_ids), 4), self.device)
        dof_pos[:, [1, 4, 7, 10]] = self.simulator.default_dof_pos[:, [1, 4, 7, 10]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)
        dof_pos[:, [2, 5, 8, 11]] = self.simulator.default_dof_pos[:, [2, 5, 8, 11]] + \
            torch_rand_float(-0.4, 0.4, (len(env_ids), 4), self.device)

        self.simulator.reset_dofs(env_ids, dof_pos, dof_vel)
    
    def _get_noise_scale_vec(self):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.  # commands
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = noise_scales.ang_vel * \
            noise_level * self.obs_scales.ang_vel
        noise_vec[9:21] = noise_scales.dof_pos * \
            noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * \
            noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.  # previous actions
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # refer to https://ieeexplore.ieee.org/document/11112615
        heading_error = torch.abs(self.heading - self.commands[:, 3])
        heading_coef = (1 + torch.cos(heading_error)) / 2
        lin_vel_x_error = torch.square(self.commands[:, 0] - self.simulator.base_lin_vel[:, 0])
        # double the weight of y velocity error to discourage y axis drifting
        lin_vel_y_error = 2 * torch.square(self.commands[:, 1] - self.simulator.base_lin_vel[:, 1])
        lin_vel_error = lin_vel_x_error + lin_vel_y_error
        # add heading_coef to make the reward smaller when the robot is facing away from the commanded direction
        # thus to encourage the robot to walk across the terrain in the commanded direction
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma) * heading_coef
    
    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        
        Attention: using torch.max(self.simulator.height_around_feet) will cause reward value jumping, bad for learning
        """
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.simulator.feet_pos[:, :, 2] - torch.max(self.simulator.height_around_feet, dim=-1)[0] -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_hip_pos(self):
        """ Reward for the hip joint position close to default position
        """
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos[:, hip_joint_indices] - 
            self.simulator.default_dof_pos[:, hip_joint_indices]), dim=-1)
        return dof_pos_error
    
    def _reward_feet_near_edge(self):
        """ Penalize feet being too close to the edge of a terrain
        """
        feet_near_edge = self.simulator.calc_feet_near_edge()
        feet_contact = self.feet_max_force_z > 10.0
        # print(f"feet near edge: {torch.sum(feet_near_edge)}")
        return torch.sum(feet_near_edge * feet_contact, dim=-1)
    