from legged_gym.envs.base.legged_robot import *
from collections import deque

class LeggedRobotTSDepth(LeggedRobot):
    
    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                               # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,          # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,                                        # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,               # num_dofs
            self.actions                                                    # num_actions
        ), dim=-1)
        
        # Domain Randomization info
        domain_randomization_info = torch.cat((
                    self.simulator._friction_values,            # 1
                    self.simulator._added_base_mass,        # 1
                    self.simulator._base_com_bias,          # 3
                    self.simulator._rand_push_vels,         # 3
            ), dim=-1)
        
        # Critic observation
        critic_obs = torch.cat((
            self.obs_buf,                 # num_observations
            domain_randomization_info,    # 35
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,    # 3
        ), dim=-1)
        if self.cfg.asset.obtain_link_contact_states:
            critic_obs = torch.cat(
                (   critic_obs,                          # previous
                    self.simulator.link_contact_states,  # contact states of hips, thighs, calfs, feet and base (4+4+4+4+1)=17
                ),
                dim=-1,
            )

        heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
            1) - 0.3 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        critic_obs = torch.cat((critic_obs, heights), dim=-1) # add height measurements
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
                    domain_randomization_info,                       # 35
                    heights,                                         # height measurements 144
                    self.simulator.base_lin_vel * self.obs_scales.lin_vel,    # 3
            ),
                dim=-1,
        )
        if self.cfg.asset.obtain_link_contact_states:
            self.privileged_obs_buf = torch.cat(
                (
                    self.privileged_obs_buf,                
                    self.simulator.link_contact_states,        # contact states of thighs, calfs and feet (4+4+4)=12
                ),dim=-1,)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions = self._pre_sim_step(actions)
        self.simulator.step(actions)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, \
            self.simulator.depth_images * self.obs_scales.depth_image, self.critic_obs_buf, \
            self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, depth_image_features, critic_obs, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs, depth_image_features, critic_obs

    def get_observations(self):
        return self.obs_buf, self.privileged_obs_buf, \
            self.simulator.depth_images * self.obs_scales.depth_image, self.critic_obs_buf

    def _init_buffers(self):
        super()._init_buffers()
        # critic observation buffer
        self.critic_obs_buf = torch.zeros(
            (self.num_envs, self.cfg.env.num_critic_obs),
            dtype=torch.float,
            device=self.device,
        )
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

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        # # log additional curriculum info
        # if self.cfg.terrain.curriculum:
        #     self.extras["episode"]["slope_level"] = torch.mean(
        #         self.simulator.terrain_levels[self.simulator.slope_env_ids].float())
        #     self.extras["episode"]["stairs_level"] = torch.mean(
        #         self.simulator.terrain_levels[self.simulator.stairs_env_ids].float())
        #     self.extras["episode"]["discrete_level"] = torch.mean(
        #         self.simulator.terrain_levels[self.simulator.discrete_env_ids].float())
        #     self.extras["episode"]["stepping_stones_level"] = torch.mean(
        #         self.simulator.terrain_levels[self.simulator.stepping_stones_env_ids].float())
        #     self.extras["episode"]["gap_level"] = torch.mean(
        #         self.simulator.terrain_levels[self.simulator.gap_env_ids].float())
        #     self.extras["episode"]["pit_level"] = torch.mean(
        #         self.simulator.terrain_levels[self.simulator.pit_env_ids].float())
        # clear obs history for the envs that are reset
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
        
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_latent_dims = self.cfg.env.num_latent_dims
        self.num_critic_obs = self.cfg.env.num_critic_obs
        if self.cfg.sensor.add_depth:
            self.depth_image_features_shape = [self.cfg.sensor.depth_camera_config.num_history,
                                              self.cfg.sensor.depth_camera_config.resolution[0],
                                              self.cfg.sensor.depth_camera_config.resolution[1]]
            self.depth_image_resolution = self.cfg.sensor.depth_camera_config.resolution
        self.num_student = self.cfg.env.num_camera_envs