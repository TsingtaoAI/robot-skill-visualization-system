from legged_gym import *

import torch

from legged_gym.envs.base.legged_robot_ts import LeggedRobotTS
from legged_gym.utils.math_utils import wrap_to_pi, quat_apply, torch_rand_float
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.constraint_manager import ConstraintManager
from .go2_cat_config import Go2CaTCfg

class Go2CaT(LeggedRobotTS):
    def __init__(self, cfg: Go2CaTCfg, sim_params, sim_device, headless):
        super().__init__(cfg, sim_params, sim_device, headless)
        if self.cfg.constraints.enable == "cat":
            self.init_done = False
            self._prepare_constraints()
            self.init_done = True
            
    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,                     # 3
            self.simulator.projected_gravity,                                         # 3
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,                   # 3
            (self.simulator.dof_pos - self.simulator.default_dof_pos) *
            self.obs_scales.dof_pos,  # num_dofs
            self.simulator.dof_vel * self.obs_scales.dof_vel,                         # num_dofs
            self.actions                                                    # num_actions
        ), dim=-1)
        
        domain_randomization_info = torch.cat((
                    self.simulator._friction_values,            # 1
                    self.simulator._added_base_mass,        # 1
                    self.simulator._base_com_bias,          # 3
                    self.simulator._rand_push_vels[:, :2],  # 2
                    self.simulator._kp_scale,                 # num_actions
                    self.simulator._kd_scale,                 # num_actions
                    self.simulator._joint_armature,         # 1
                    self.simulator._joint_friction,        # 1
                    self.simulator._joint_damping,          # 1
            ), dim=-1)
        
        # Critic observation
        critic_obs = torch.cat((
            self.obs_buf,                 # num_observations
            domain_randomization_info,    # 34
        ), dim=-1)
        if self.cfg.asset.obtain_link_contact_states:
            critic_obs = torch.cat(
                (
                    critic_obs,                         # previous
                    self.simulator.link_contact_states,  # contact states of thighs, calfs and feet (4+4+4)=12
                ),
                dim=-1,
            )
        if self.cfg.terrain.measure_heights: # 81
            heights = torch.clip(self.simulator.base_pos[:, 2].unsqueeze(
                1) - 0.5 - self.simulator.measured_heights, -1, 1.) * self.obs_scales.height_measurements
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

        # push obs_buf to obs_history
        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
        
        # Privileged observation, for privileged encoder
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                (
                    domain_randomization_info,                       # 34
                    self.simulator.height_around_feet.flatten(1,2),  # 9*number of feet
                    self.simulator.normal_vector_around_feet,        # 3*number of feet
                ),
                dim=-1,
            )
            if self.cfg.asset.obtain_link_contact_states:
                self.privileged_obs_buf = torch.cat(
                    (
                        self.privileged_obs_buf,                   # previous
                        self.simulator.link_contact_states,        # contact states of thighs, calfs and feet (4+4+4)=12
                    ),
                    dim=-1,
                )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.cfg.constraints.enable == "cat":
            self.extras["episode"]["cstr_probs"] = torch.mean(self.cstr_prob)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self.simulator.draw_debug_vis() if needed
        """
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.simulator.post_physics_step()
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        if self.cfg.constraints.enable == "cat":
            self.compute_constraints_cat()
        self.compute_reward()
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.llast_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        
        if self.debug:
            self.simulator.draw_debug_vis()
    
    def compute_constraints_cat(self):
        """Compute various constraints for constraints as terminations. Constraints violations are asssessed then
        handed out to a constraint manager (ConstraintManager class) that will compute termination probabilities."""
        if not self.init_done:
            return
        # ------------ Soft constraints ----------------
        
        # Torque constraint
        cstr_torque = torch.any(torch.abs(self.simulator.torques) > self.simulator.torque_limits, dim=-1)
        
        # Joint velocity constraint
        cstr_dof_vel = torch.any(torch.abs(self.simulator.dof_vel) > self.simulator.dof_vel_limits, dim=-1)
        
        # Action rate constraint (for command smoothness)
        cstr_action_rate = torch.any(torch.abs(self.actions - self.last_actions) / self.dt > 
                                     self.cfg.constraints.limits.action_rate, dim=-1)
        
        # Base height constraint (too low)
        cstr_base_height = torch.mean(self.simulator.base_pos[:, 2].unsqueeze(
            1) - self.simulator.measured_heights, dim=1) < self.cfg.constraints.limits.min_base_height

        # ------------ Hard constraints ----------------
        
        # Collision constraint
        cstr_collision = torch.any(self.penalized_bodies_force_norm > 10.0, dim=1)
        
        # Feet stumble constraint
        # if the dim of link_contact_forces is 4. It has history, shape is (num_envs, history_length, num_links, 3)
        if len(self.simulator.link_contact_forces.shape) == 4:
            feet_max_norm_xy = torch.max(torch.norm(self.simulator.link_contact_forces[:, :, self.simulator.feet_contact_indices, :2], dim=-1), dim=1)[0]
            feet_max_force_z = torch.max(torch.abs(self.simulator.link_contact_forces[:, :, self.simulator.feet_contact_indices, 2]), dim=1)[0]
        else:
            feet_max_norm_xy = torch.norm(self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, :2], dim=-1)
            feet_max_force_z = torch.abs(self.simulator.link_contact_forces[:, self.simulator.feet_contact_indices, 2])

        cstr_feet_stumble = torch.any(feet_max_norm_xy > 4 * feet_max_force_z, dim=1)

        # Joint position limit constraint
        cstr_dof_pos = torch.any(self.simulator.dof_pos < self.simulator.dof_pos_limits[:, 0], dim=-1) * \
            torch.any(self.simulator.dof_pos > self.simulator.dof_pos_limits[:, 1], dim=-1)
        
        # Base orientation constraint
        cstr_base_orientation = self.simulator.projected_gravity[:, 2] > self.cfg.constraints.limits.max_projected_gravity
        
        # ------------ Style constraints ----------------
        
        # Standing still constraint, penalize motion when command is zero
        cstr_stand_still = torch.any(torch.abs(self.simulator.dof_vel) > 4.0, dim=-1) * \
            (torch.norm(self.commands[:, :3], dim=1) < 0.1).float().unsqueeze(1)
        
        # ------------ Log constraint violation ----------------
        if self.debug_cstr:
            cstr_names = ["torque", "dof_vel", "action_rate", "base_height",
                            "collision", "feet_stumble", "dof_pos", "base_orientation",
                            "stand_still"]
            cstr_violations = [cstr_torque, cstr_dof_vel, cstr_action_rate, cstr_base_height,
                                cstr_collision, cstr_feet_stumble, cstr_dof_pos, cstr_base_orientation,
                                cstr_stand_still]
            for i in range(len(cstr_names)):
                name = cstr_names[i]
                if name not in self.cstr_violation:
                    self.cstr_violation[name] = 0
                violation = cstr_violations[i]
                self.cstr_violation[name] += torch.mean(violation.float()).item()
        
        # ------------ Applying constraints ----------------
        
        soft_p = self.constraint["soft_p"]
        
        # soft constraints
        self.constraint_manager.add("torque", cstr_torque, max_p=soft_p)
        self.constraint_manager.add("dof_vel", cstr_dof_vel, max_p=soft_p)
        self.constraint_manager.add("action_rate", cstr_action_rate, max_p=soft_p)
        self.constraint_manager.add("base_height", cstr_base_height, max_p=soft_p)
        # hard constraints
        self.constraint_manager.add("collision", cstr_collision, max_p=1.0)
        self.constraint_manager.add("feet_stumble", cstr_feet_stumble, max_p=1.0)
        self.constraint_manager.add("dof_pos", cstr_dof_pos, max_p=1.0)
        self.constraint_manager.add("base_orientation", cstr_base_orientation, max_p=1.0)
        # style constraints
        self.constraint_manager.add("stand_still", cstr_stand_still, max_p=soft_p)
        
        self.constraint_manager.log_all(self.episode_sums)
        
        # Get final termination probability for each env from all constraints
        self.cstr_prob = self.constraint_manager.get_probs()
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            if self.cfg.constraints.enable == "cat":
                self.rew_buf[:] = torch.clip(self.rew_buf[:] * (1.0 - self.cstr_prob), min=0.)
            else:
                self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination(
            ) * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    #---------- Protected Functions ----------#
    
    def _prepare_constraints(self):
        self.constraint = {}
        self.constraint["soft_p"] = self.cfg.constraints.soft_p
        self.constraint_manager = ConstraintManager(tau=self.cfg.constraints.tau_constraint, min_p=0.0)
        
    def _init_buffers(self):
        super()._init_buffers()
        # constraint probabilities buffer
        self.cstr_prob = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    
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

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        # if debug_cstr_violation exists in cfg, use it; otherwise, set to False
        if hasattr(self.cfg.env, 'debug_cstr_violation'):
            self.debug_cstr = self.cfg.env.debug_cstr_violation
            self.cstr_violation = {}
        else:
            self.debug_cstr = False
    
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
        return noise_vec
    
    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.feet_force_norm > 10.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.25) * first_contact, dim=1)  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_foot_clearance(self):
        """
        Encourage feet to be close to desired height while swinging
        
        Attention: using torch.max(self.simulator.height_around_feet) will cause reward value jumping, bad for learning
        """
        foot_vel_xy_norm = torch.norm(self.simulator.feet_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(
                self.simulator.feet_pos[:, :, 2] - torch.mean(self.simulator.height_around_feet, dim=-1) -
                self.cfg.rewards.foot_clearance_target -
                self.cfg.rewards.foot_height_offset
            ), dim=-1
        )
        return torch.exp(-clearance_error / self.cfg.rewards.foot_clearance_tracking_sigma)
    
    def _reward_hip_pos(self):
        """ Reward for the hip joint position close to default position,
        """
        hip_joint_indices = [0, 3, 6, 9]
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos[:, hip_joint_indices] - 
            self.simulator.default_dof_pos[:, hip_joint_indices]), dim=-1)
        return dof_pos_error