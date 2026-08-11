from legged_gym.envs.base.legged_robot import *
from collections import deque

class LeggedRobotEE(LeggedRobot):
    
    def compute_observations(self):
        obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,  # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions                                                    # num_actions
        ), dim=-1)
        
        # Estimator labels
        self.estimator_labels_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,         # 3
            self.simulator.link_contact_states, # contact states of hips, thighs, calfs, feet and base (4+4+4+4+1)=17
            torch.clip(self.simulator.feet_pos[:, :, 2] -
            torch.mean(self.simulator.height_around_feet, dim=-1) -
            self.cfg.rewards.foot_height_offset, -1, 1.),  # 4
        ), dim=-1)
        
        # Critic observation
        single_critic_obs = torch.cat((
            obs_buf,                 # num_observations
            self.estimator_labels_buf,    # 24
        ), dim=-1)
        
        if self.cfg.terrain.measure_heights: # 81
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            single_critic_obs = torch.cat((single_critic_obs, heights), dim=-1)
        self.critic_obs_deque.append(single_critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) -
                             1) * self.noise_scale_vec

        # push obs_buf to obs_history
        self.obs_history_deque.append(obs_buf)
        self.estimator_features_buf = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
    
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
        self.estimator_features_buf = torch.clip(self.estimator_features_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.estimator_features_buf, self.estimator_labels_buf, self.privileged_obs_buf, \
            self.rew_buf, self.reset_buf, self.extras
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        estimator_features, estimator_labels, privileged_obs, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return estimator_features, estimator_labels, privileged_obs
    
    def get_observations(self):
        # estimator input, estimator target, critic obs
        return self.estimator_features_buf, self.estimator_labels_buf, self.privileged_obs_buf
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_estimator_features = cfg.env.num_estimator_features
        self.num_estimator_labels = cfg.env.num_estimator_labels
    
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
        
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0