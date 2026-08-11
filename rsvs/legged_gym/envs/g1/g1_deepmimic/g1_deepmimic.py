from legged_gym import *
import numpy as np
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import *
from legged_gym.utils.motion_loader import MotionLoader
from collections import deque

class G1DeepMimic(LeggedRobot):
    
    def compute_observations(self):
        key_body_pos_relative_to_base = self.simulator.key_body_pos - \
                self.simulator.base_pos.unsqueeze(1)

        # get key body point position relative to base in the body frame
        key_body_pos_b = quat_rotate_inverse(
            self.simulator.base_quat.unsqueeze(1).repeat(1, self.simulator.key_body_pos.shape[1], 1),
            key_body_pos_relative_to_base 
        )
        ref_motion_obs = self._get_ref_motion_obs()
        
        obs_buf = torch.cat((
            # proprioceptive features
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.base_quat,
            (self.simulator.dof_pos - 
             self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            key_body_pos_b.flatten(start_dim=1),
            self.actions,
            # reference motion features
            ref_motion_obs,
        ), dim=-1)
        
        # domain randomization params
        domain_params = torch.cat((
            self.simulator.dr_friction_values,
            self.simulator.dr_added_base_mass,
            self.simulator.dr_base_com_bias,
            self.simulator.dr_rand_push_vels,
        ), dim=-1)
        
        single_critic_obs = torch.cat((
            (self.simulator.base_pos - self.simulator.env_origins),
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self.motion_loader.get_ref_base_pos(),
            obs_buf,
            domain_params,
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
        
    def post_physics_step(self):
        ref_time_out_env_ids = self.motion_loader.step_frame_index()
        # reset the envs that have reached the end of the reference motion trajectory
        if len(ref_time_out_env_ids) > 0:
            # BUG: IsaacGym requires 1 step after resetting to get the correct rigid body states
            # When enabling reference motion termination (env terminate when the distance between the robot and the reference motion is too large),
            # the rigid body state does not update after this reset, which causes the termination abnormally.
            # The dof state and root state is reset correctly, but the rigid body state is not updated
            _ = np.random.random()
            if self.cfg.init_state.reference_state_initialization and _ < self.cfg.init_state.reference_state_initialization_prob:
                self._reset_dofs_from_reference_motion(ref_time_out_env_ids)
                self._reset_root_states_from_reference_motion(ref_time_out_env_ids)
            else:
                self._reset_root_states(ref_time_out_env_ids)
                self._reset_dofs(ref_time_out_env_ids)
        super().post_physics_step()
        if self.debug:
            ref_key_body_pos = self.motion_loader.get_ref_key_body_pos() \
                + self.motion_loader.get_ref_base_pos().unsqueeze(1) \
                + self.simulator.env_origins.unsqueeze(1)
            self.simulator.draw_debug_vis(ref_key_body_pos)
            
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length ==0):
            self._update_command_curriculum(env_ids)

        self.motion_loader.resample_frame_index(env_ids)
        self._resample_commands(env_ids)
        _ = np.random.random()
        if self.cfg.init_state.reference_state_initialization and _ < self.cfg.init_state.reference_state_initialization_prob:
            self._reset_dofs_from_reference_motion(env_ids)
            self._reset_root_states_from_reference_motion(env_ids)
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
    
    def _reset_dofs_from_reference_motion(self, env_ids):
        # reset dofs to match the reference motion at the current frame index
        dof_pos = self.motion_loader.get_ref_dof_pos(env_ids)
        dof_vel = self.motion_loader.get_ref_dof_vel(env_ids)
        self.simulator.reset_dofs(env_ids, 
                                  dof_pos, 
                                  dof_vel)

    def _reset_root_states_from_reference_motion(self, env_ids):
        # reset root states to match the reference motion at the current frame index
        
        root_pos = self.motion_loader.get_ref_base_pos(env_ids) + self.simulator.env_origins[env_ids]
        root_pos[:, 2] = self.simulator.base_init_pos[2] # set the vertical position to the default value to avoid initial penetration
        root_rot = self.motion_loader.get_ref_base_quat(env_ids)
        root_lin_vel = self.motion_loader.get_ref_base_lin_vel(env_ids)
        root_ang_vel = self.motion_loader.get_ref_base_ang_vel(env_ids)
        self.simulator.reset_root_states(env_ids, 
                                         root_pos, 
                                         root_rot, 
                                         root_lin_vel, 
                                         root_ang_vel)
    
    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[:3] = 0.
        # noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel # ang_vel
        # noise_vec[6:9] = noise_scales.gravity * noise_level # projected gravity
        # noise_vec[9:9 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # noise_vec[9 + self.num_actions:9 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # noise_vec[9 + 2 * self.num_actions:24 + 2*self.num_actions] = 0.  # key body pos relative to base and actions
        # noise_vec[24 + 2*self.num_actions:24 + 3 * self.num_actions] = 0.  # previous actions
        # noise_vec[24 + 3 * self.num_actions:25 + 3 * self.num_actions] = 0. # frame_portion
        # noise_vec[25 + 3 * self.num_actions:28 + 3 * self.num_actions] = 0.  # ref_root_lin_vel
        # noise_vec[28 + 3 * self.num_actions:31 + 3 * self.num_actions] = 0.  # ref_root_ang_vel
        # noise_vec[31 + 3 * self.num_actions:34 + 3 * self.num_actions] = 0.  # ref_projected_gravity
        # noise_vec[34 + 3 * self.num_actions:34 + 4 * self.num_actions] = 0.  # ref_dof_pos
        # noise_vec[34 + 4 * self.num_actions:34 + 5 * self.num_actions] = 0.  # ref_dof_vel
        # noise_vec[34 + 5 * self.num_actions:49 + 5 * self.num_actions] = 0.  # ref_key_body_pos_relative_to_base
        return noise_vec
    
    def _init_buffers(self):
        super()._init_buffers()
        self.motion_loader = MotionLoader(
            self.num_envs,
            self.dt,
            self.cfg.env.motion_file,
            self.device
        )
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
    
    def _get_ref_motion_obs(self):
        """Get the reference motion features for the current frame index.
        """
        ref_motion_obs = []
        current_frame_idx = self.motion_loader.frame_index
        for i in range(self.cfg.env.ref_motion_frame_stack):
            frame_idx = current_frame_idx + i
            over_length_env_ids = (frame_idx >= self.motion_loader.trajectory_num_frames).nonzero(as_tuple=False).flatten()
            frame_idx[over_length_env_ids] = frame_idx[over_length_env_ids] % self.motion_loader.trajectory_num_frames
            ref_motion_obs.append(self._get_single_frame_ref_motion_obs(frame_idx))
        ref_motion_obs = torch.cat(ref_motion_obs, dim=-1)
        return ref_motion_obs
    
    def _get_single_frame_ref_motion_obs(self, frame_idx):
        """Get the reference motion features for the given frame index.
        """
        ref_base_quat = self.motion_loader.get_ref_base_quat_at_idx(frame_idx)
        ref_key_body_pos_relative_to_base = self.motion_loader.get_ref_key_body_pos_at_idx(frame_idx)
        key_body_pos_b = quat_rotate_inverse(
            # repeat the quaternion for each key body point, [N,4]->[N,num_key_bodies,4]
            ref_base_quat.unsqueeze(1).repeat(1, ref_key_body_pos_relative_to_base.shape[1], 1),
            ref_key_body_pos_relative_to_base # [N,num_key_bodies,3] 
        )
        ref_base_lin_vel_b = quat_rotate_inverse(
            ref_base_quat,
            self.motion_loader.get_ref_base_lin_vel_at_idx(frame_idx)
        )
        ref_base_ang_vel_b = quat_rotate_inverse(
            ref_base_quat,
            self.motion_loader.get_ref_base_ang_vel_at_idx(frame_idx)
        )
        ref_motion_obs = torch.cat((
            ref_base_lin_vel_b * self.obs_scales.lin_vel,
            ref_base_ang_vel_b * self.obs_scales.ang_vel,
            ref_base_quat,
            (self.motion_loader.get_ref_dof_pos_at_idx(frame_idx) - 
             self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.motion_loader.get_ref_dof_vel_at_idx(frame_idx) * self.obs_scales.dof_vel,
            key_body_pos_b.flatten(start_dim=1),
        ), dim=-1)
        
        return ref_motion_obs
        
    def _reward_tracking_ref_dof_pos(self):
        """Reward term for imitating the reference motion's dof positions.
        """
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos - 
            self.motion_loader.get_ref_dof_pos()), dim=-1)
        
        return torch.exp(-dof_pos_error 
                         / self.cfg.rewards.tracking_dof_pos_sigma)
    
    def _reward_tracking_ref_dof_vel(self):
        """Reward term for imitating the reference motion's dof velocities.
        """
        dof_vel_error = torch.sum(torch.abs(
            self.simulator.dof_vel - 
            self.motion_loader.get_ref_dof_vel()), dim=-1)
        
        return torch.exp(-dof_vel_error 
                         / self.cfg.rewards.tracking_dof_vel_sigma)
        
    def _reward_tracking_ref_base_pose(self):
        """Reward term for imitating the reference motion's base position and orientation.
        """
        base_pos_error = torch.sum(torch.square(
            self.simulator.base_pos - 
            (self.motion_loader.get_ref_base_pos() + 
             self.simulator.env_origins)), dim=-1)
        
        base_rot_error = torch.sum(torch.square(
            self.simulator.base_quat - 
            self.motion_loader.get_ref_base_quat()), dim=-1)
        
        return torch.exp(-(base_pos_error + 0.1 * base_rot_error) /
                         self.cfg.rewards.tracking_ref_base_pose_sigma)
    
    def _reward_tracking_ref_base_vel(self):
        """Reward term for imitating the reference motion's root linear velocity and root angular velocity.
        """
        ref_base_quat = self.motion_loader.get_ref_base_quat()
        ref_base_lin_vel_b = quat_rotate_inverse(
            ref_base_quat,
            self.motion_loader.get_ref_base_lin_vel()
        )
        base_lin_vel_error = torch.sum(torch.square(
            self.simulator.base_lin_vel - 
            ref_base_lin_vel_b), dim=-1)
        
        ref_base_ang_vel_b = quat_rotate_inverse(
            ref_base_quat,
            self.motion_loader.get_ref_base_ang_vel()
        )
        base_ang_vel_error = torch.sum(torch.square(
            self.simulator.base_ang_vel - 
            ref_base_ang_vel_b), dim=-1)
        
        return torch.exp(-(base_lin_vel_error + 0.1 * base_ang_vel_error) /
                         self.cfg.rewards.tracking_ref_base_vel_sigma)
    
    def _reward_tracking_ref_key_pos(self):
        """Reward term for imitating the reference motion's key body position relative to base.
        """
        key_body_pos_relative_to_base = self.simulator.key_body_pos - \
                self.simulator.base_pos.unsqueeze(1)
        key_body_pos_error = torch.sum(torch.square(
            key_body_pos_relative_to_base - 
            self.motion_loader.get_ref_key_body_pos()), dim=[1,2])
        
        return torch.exp(-key_body_pos_error / 
                         self.cfg.rewards.tracking_ref_key_pos_sigma)