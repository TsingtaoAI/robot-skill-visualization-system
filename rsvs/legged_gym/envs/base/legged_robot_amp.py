from legged_gym.envs.base.legged_robot import *
from legged_gym.utils.motion_loader import AMPLoader

class LeggedRobotAMP(LeggedRobot):
    
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
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras,\
            self.reset_env_ids, self.terminal_amp_states
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _, _ = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self._update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        self.reset_env_ids = env_ids
        self.terminal_amp_states = self.get_amp_observations()[env_ids]
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length ==0):
            self._update_command_curriculum(env_ids)

        self._resample_commands(env_ids)
        _ = np.random.random()
        if self.cfg.init_state.reference_state_initialization \
            and _ < self.cfg.init_state.reference_state_initialization_prob:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_from_reference_motion(env_ids, frames)
            self._reset_root_states_from_reference_motion(env_ids, frames)
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

    def get_amp_observations(self):
        key_body_pos_relative_to_base = self.simulator.key_body_pos - \
                self.simulator.base_pos.unsqueeze(1)
        # Use base_lin_vel_w, base_ang_vel_w, dof_pos, dof_vel, key_body_pos_relative_to_base in the observations
        return torch.cat((
            self.simulator.base_lin_vel,              # 3
             self.simulator.base_ang_vel,             # 3
            self.simulator.dof_pos,                   # num_dofs
            self.simulator.dof_vel,                   # num_dofs
            key_body_pos_relative_to_base.flatten(start_dim=1), # num_key_bodies * 3
        ), dim=-1)
        
    def _init_buffers(self):
        super()._init_buffers()
        self.reset_env_ids = None
        self.terminal_amp_states = None
        if self.cfg.init_state.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, 
                                        device=self.device, 
                                        time_between_frames=self.dt,
                                        num_dof=self.num_actions,
                                        num_key_bodies=len(self.simulator.key_body_indices))
        
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