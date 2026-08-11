from abc import ABC, abstractmethod
import torch
from torch import Tensor
import numpy as np
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.math_utils import dr_normalize

""" ********** Base Simulator ********** """
class Simulator(ABC):
    def __init__(self, cfg : LeggedRobotCfg, sim_params: dict, sim_device: str = "cuda:0", headless: bool = False):
        self._height_samples = None
        self._device = sim_device
        self._headless = headless
        self._cfg = cfg
        self._num_envs = self._cfg.env.num_envs
        self._num_actions = self._cfg.env.num_actions
        self._dof_indices = []  # align joint orders in different simulators with the order specified in the config file
        self._parse_cfg()
        self._create_sim()
        self._create_envs()
        self._init_buffers()

    #----- Public methods -----#
    @abstractmethod
    def step(self):
        """Performs a simulation step, which typically includes applying actions, stepping the physics simulation, and updating states and observations.
        """
        return

    @abstractmethod
    def post_physics_step(self):
        """Performs any necessary updates after the physics step.
        """
        return
    
    @abstractmethod
    def reset_idx(self, env_ids: Tensor):
        """Reset environments with the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to reset.
        """
        return
    
    @abstractmethod
    def reset_dofs(self, env_ids: Tensor, dof_pos: Tensor, dof_vel: Tensor):
        """Reset the DOF states of the environments with the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to reset.
            dof_pos (Tensor): DOF positions to reset.
            dof_vel (Tensor): DOF velocities to reset.
        """
        return
    
    @abstractmethod
    def reset_root_states(self, 
                          env_ids: Tensor, 
                          base_pos: Tensor, 
                          base_quat: Tensor, 
                          base_lin_vel_w: Tensor, 
                          base_ang_vel_w: Tensor):
        """Reset the root states of the environments with the given environment IDs.
        
        Args:
            env_ids (Tensor): Environment IDs to reset.
            base_pos (Tensor): Base positions to reset.
            base_quat (Tensor): Base orientations (quaternions, xyzw sequence) to reset.
            base_lin_vel_w (Tensor): Base linear velocities in world frame to reset.
            base_ang_vel_w (Tensor): Base angular velocities in world frame to reset.
        """
        return
    
    @abstractmethod
    def update_terrain_curriculum(self, env_ids, move_up, move_down):
        """Updates the terrain curriculum for the environments with the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to update terrain curriculum for.
            move_up (Tensor): A boolean tensor indicating whether to move up the terrain curriculum for each environment.
            move_down (Tensor): A boolean tensor indicating whether to move down the terrain curriculum for each environment.
        """
        return
    
    @abstractmethod
    def push_robots(self):
        """Apply perturbation velocity to the base of the robot as domain randomization.
        """
        return
    
    @abstractmethod
    def push_links(self):
        """Apply perturbation forces to the links of the robot as domain randomization.
        """
        return
    
    @abstractmethod
    def draw_debug_vis(self):
        """Draws debug visualizations, such as the sampling points around the robot.
        """
        return
    
    @abstractmethod
    def set_viewer_camera(self, eye: np.ndarray, target: np.ndarray):
        """Sets the viewer camera in the simulator.

        Args:
            eye (np.ndarray): The position of the camera.
            target (np.ndarray): The target point the camera is looking at.
        """
        return
    
    @abstractmethod
    def update_sensors(self):
        """Updates the sensor readings, such as depth image sensors and lidar sensors.
        """
        return
    
    @abstractmethod
    def calc_feet_near_edge(self):
        """Calculates whether the feet are near the edge of a terrain, which is used for reward calculation.

        Returns:
            Tensor((num_envs, num_feet)): A boolean tensor indicating whether each foot is near the edge of a terrain for each environment.
        """
        return

    #----- Protected methods -----#
    @abstractmethod
    def _pre_simulator_step(self, actions):
        """Performs any necessary updates before the simulator step
        """
        return
    
    @abstractmethod
    def _parse_cfg(self):
        """Parses the configuration file and initializes necessary variables for the simulator.
        """
        return

    @abstractmethod
    def _create_sim(self):
        """Creates the simulation environment, including the physics engine and any necessary components.
        """
        return

    @abstractmethod
    def _create_envs(self):
        """Creates environments, adds the robot asset to each environment, sets DOF properties and calls callbacks to process rigid shape, rigid body and DOF properties.
        """
        return

    @abstractmethod
    def _init_buffers(self):
        """Initializes necessary buffers for the simulator, such as those for observations, actions, rewards, and any other relevant data.
        """
        return
    
    @abstractmethod
    def _init_height_points(self):
        """Initializes the height sampling points around the robot in the base frame, which are used for measuring terrain heights.
        """
        return
    
    @abstractmethod
    def _get_env_origins(self):
        """Gets the origin positions of all environments, which are used for resetting the robot to the correct height.
        """
        return
    
    @abstractmethod
    def _update_surrounding_heights(self):
        """Updates the height of the sampling points around the robot.
        
        The sampling grid is defined in LeggedRobotCfg.terrain.measured_points_x/y.
        """
        return
    
    @abstractmethod
    def _check_base_pos_out_of_bound(self):
        """Checks if the base position of the robot is out of bound for all environments.

        Returns:
            Tensor: A boolean tensor indicating whether the base position of the robot is out of bound for each environment.
        """
        return
    
    @abstractmethod
    def _compute_torques(self, actions: Tensor):
        """Computes the torques to apply to the robot's joints based on the given actions.

        Args:
            actions (Tensor): Actions to compute torques for.

        Returns:
            Tensor: Torques to apply to the robot's joints.
        """
        return
        
    
    @abstractmethod
    def _init_domain_params(self):
        """Initializes domain randomization parameters, which are used as privilege information.
        """
        return
    
    @abstractmethod
    def _randomize_friction(self, env_ids: Tensor):
        """Randomizes the friction of all links for the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to randomize friction for. If None, randomizes friction for all environments. Defaults to None.
        """
        return

    @abstractmethod
    def _randomize_restitution(self, env_ids: Tensor):
        """Randomizes the restitution of all links for the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to randomize restitution for.
        """
        return

    @abstractmethod
    def _randomize_base_mass(self, env_ids: Tensor):
        """Randomizes the base mass of the robot for the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to randomize base mass for. If None, randomizes base mass for all environments. Defaults to None.
        """
        return
    
    @abstractmethod
    def _randomize_com_displacement(self, env_ids: Tensor):
        """Randomizes the center of mass (COM) displacement of the robot for the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to randomize COM displacement for. If None, randomizes COM displacement for all environments. Defaults to None.
        """
        return
    
    @abstractmethod
    def _randomize_joint_armature(self, env_ids: Tensor):
        """Randomizes the joint armature of the robot for the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to randomize joint armature for. If None, randomizes joint armature for all environments. Defaults to None.
        """
        return
    
    @abstractmethod
    def _randomize_joint_friction(self, env_ids: Tensor):
        """Randomizes the joint friction of the robot for the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to randomize joint friction for. If None, randomizes joint friction for all environments. Defaults to None.
        """
        return
    
    @abstractmethod
    def _randomize_joint_damping(self, env_ids: Tensor):
        """Randomizes the joint damping of the robot for the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to randomize joint damping for. If None, randomizes joint damping for all environments. Defaults to None.
        """
        return
    
    @abstractmethod
    def _randomize_pd_gain(self, env_ids: Tensor):
        """Randomizes the PD gain of the robot for the given environment IDs.

        Args:
            env_ids (Tensor): Environment IDs to randomize PD gain for. If None, randomizes PD gain for all environments. Defaults to None.
        """
        return
    
    @abstractmethod
    def _update_depth_camera(self):
        """Updates the depth camera readings
        """
        return
    
    #----- Properties -----#
    @property
    def feet_indices(self):
        """Returns the indices of the feet links in the robot articulation.

        Returns:
            list[int]: Indices of the feet links.
        """
        return self._feet_indices
    
    @property
    def feet_contact_indices(self):
        """Returns the indices of the feet links in the contact sensors.
        
            This property is created to solve the difference of body orders between
            articulation and contact sensors in IsaacLab

        Returns:
            list[int]: Indices of the feet links in the contact sensors.
        """
        return self._feet_contact_indices
    
    @property
    def termination_contact_indices(self):
        """Returns the indices of the links in the contact sensors that are used for termination checking.

        Returns:
            list[int]: Indices of the links in the contact sensors for termination checking.
        """
        return self._termination_contact_indices
    
    @property
    def penalized_contact_indices(self):
        """Returns the indices of the links in the contact sensors that are used for penalty.

        Returns:
            list[int]: Indices of the links in the contact sensors for penalty.
        """
        return self._penalized_contact_indices
    
    @property
    def terrain_types(self):
        """Returns the terrain types of all environments.

        Returns:
            Tensor((num_envs,)): Terrain types of all environments.
        """
        return self._terrain_types
    
    @property
    def terrain_levels(self):
        """Returns the terrain levels of all environments.

        Returns:
            Tensor((num_envs,)): Terrain levels of all environments.
        """
        return self._terrain_levels
    
    @property
    def dof_pos_limits(self):
        """Returns the DOF position limits of the robot.

        Returns:
            Tensor((num_dof, 2)): DOF position limits of the robot.
        """
        return self._dof_pos_limits
    
    @property
    def dof_vel_limits(self):
        """Returns the DOF velocity limits of the robot.

        Returns:
            Tensor((num_dof,)): DOF velocity limits of the robot.
        """
        return self._dof_vel_limits
    
    @property
    def base_init_pos(self):
        """Returns the initial base position of the robot.

        Returns:
            Tensor((3,)): Initial base position of the robot.
        """
        return self._base_init_pos
    
    @property
    def base_init_quat(self):
        """Returns the initial base orientation (quaternion) of the robot (xyzw sequence).

        Returns:
            Tensor((4,)): Initial base orientation (quaternion) of the robot (xyzw sequence).
        """
        return self._base_init_quat
    
    @property
    def base_lin_vel(self):
        """Returns the base linear velocity of the robot (respective to the base frame).

        Returns:
            Tensor((num_envs, 3)): Base linear velocity of the robot.
        """
        return self._base_lin_vel
    
    @property
    def last_base_lin_vel(self):
        """Returns the base linear velocity of the robot (respective to the base frame) in the last control step.

        Returns:
            Tensor((num_envs, 3)): Base linear velocity of the robot in the last control step.
        """
        return self._last_base_lin_vel
    
    @property
    def base_ang_vel(self):
        """Returns the base angular velocity of the robot (respective to the base frame).

        Returns:
            Tensor((num_envs, 3)): Base angular velocity of the robot.
        """
        return self._base_ang_vel
    
    @property
    def last_base_ang_vel(self):
        """Returns the base angular velocity of the robot (respective to the base frame) in the last control step

        Returns:
            Tensor((num_envs, 3)): Base angular velocity of the robot.
        """
        return self._last_base_ang_vel
    
    @property
    def projected_gravity(self):
        """Returns the projected gravity in the base frame.

        Returns:
            Tensor((num_envs, 3)): Projected gravity in the base frame.
        """
        return self._projected_gravity
    
    @property
    def dof_pos(self):
        """Returns the DOF positions of the robot.

        Returns:
            Tensor((num_envs, num_dof)): DOF positions of the robot.
        """
        return self._dof_pos
    
    @property
    def dof_vel(self):
        """Returns the DOF velocities of the robot.

        Returns:
            Tensor((num_envs, num_dof)): DOF velocities of the robot.
        """
        return self._dof_vel
    
    @property
    def last_dof_vel(self):
        """Returns the DOF velocities of the robot in the last simulation step.

        Returns:
            Tensor((num_envs, num_dof)): DOF velocities of the robot in the last simulation step.
        """
        return self._last_dof_vel
    
    @property
    def feet_pos(self):
        """Returns the positions of the feet in the world frame.

        Returns:
            Tensor((num_envs, num_feet, 3)): Positions of the feet in the world frame.
        """
        return self._feet_pos
    
    @property
    def key_body_indices(self):
        """Returns the indices of the key bodies in the robot articulation.

        Returns:
            list[int]: Indices of the key bodies.
        """
        return self._key_body_indices
    
    @property
    def key_body_pos(self):
        """Returns the positions of the key bodies in the world frame.

        Returns:
            Tensor((num_envs, num_key_bodies, 3)): Positions of the key bodies in the world frame.
        """
        return self._key_body_pos
    
    @property
    def feet_vel(self):
        """Returns the velocities of the feet in the world frame.

        Returns:
            Tensor((num_envs, num_feet, 3)): Velocities of the feet in the world frame.
        """
        return self._feet_vel
    
    @property
    def feet_quat(self):
        """Returns the orientations (quaternions) of the feet in the world frame (xyzw sequence).

        Returns:
            Tensor((num_envs, num_feet, 4)): Orientations (quaternions) of the feet in the world frame (xyzw sequence).
        """
        return self._feet_quat
    
    @property
    def last_feet_vel(self):
        """Returns the velocities of the feet in the world frame in the last simulation step.

        Returns:
            Tensor((num_envs, num_feet, 3)): Velocities of the feet in the world frame in the last simulation step.
        """
        return self._last_feet_vel
    
    @property
    def base_pos(self):
        """Returns the base position of the robot in the world frame.

        Returns:
            Tensor((num_envs, 3)): Base position of the robot in the world frame.
        """
        return self._base_pos
    
    @property
    def base_quat(self):
        """Returns the base orientation (quaternion) of the robot in the world frame (xyzw sequence).

        Returns:
            Tensor((num_envs, 4)): Base orientation (quaternion) of the robot in the world frame (xyzw sequence).
        """
        return self._base_quat
    
    @property
    def base_euler(self):
        """Returns the base orientation (euler angles) of the robot in the world frame.

        Returns:
            Tensor((num_envs, 3)): Base orientation (euler angles) of the robot in the world frame.
        """
        return self._base_euler
    
    @property
    def measured_heights(self):
        """Returns the measured heights of the sampling points around the robot.

        Returns:
            Tensor((num_envs, num_height_points)): Measured heights of the sampling points around the robot.
        """
        return self._measured_heights
    
    @property
    def link_contact_forces(self):
        """Returns the contact forces of all links of the robot.

        Returns:
            Tensor((num_envs, num_links, 3)): Contact forces of all links of the robot.
        """
        return self._link_contact_forces
    
    @property
    def link_contact_states(self):
        """Returns the contact states of specified links of the robot.

        Returns:
            Tensor((num_envs, num_links)): Contact states of specified links of the robot.
        """
        return self._link_contact_states
    
    @property
    def torques(self):
        """Returns the torques applied to the robot's joints.

        Returns:
            Tensor((num_envs, num_dof)): Torques applied to the robot's joints.
        """
        return self._torques
    
    @property
    def torque_limits(self):
        """Returns the torque limits of the robot's joints.

        Returns:
            Tensor((num_dof)): Torque limits of the robot's joints.
        """
        return self._torque_limits
    
    @property
    def normal_vector_around_feet(self):
        """Returns the terrain normal vectors around feet.
        
        Returns:
            Tensor((num_envs, num_feet * 3)): Terrain normal vectors around feet.
        """
        return self._normal_vector_around_feet
    
    @property
    def height_around_feet(self):
        """Returns the terrain heights around feet.
        
        Returns:
            Tensor((num_envs, num_feet, 9)): Terrain heights around feet.
        """
        return self._height_around_feet
    
    @property
    def default_dof_pos(self):
        """Returns the default dof pos.
        
        Returns:
            Tensor((1, num_dofs)): Default dof pos.
        """
        return self._default_dof_pos
    
    @property
    def custom_origins(self):
        """Returns whether the environments use custom origins. 

        Returns:
            bool: Whether the environments use custom origins.
        """
        return self._custom_origins
    
    @property
    def dr_friction_values(self):
        """Returns the friction values for domain randomization.

        Returns:
            Tensor((num_envs, 1)): Friction values for domain randomization.
        """
        return dr_normalize(self._friction_values, 
                            self._cfg.domain_rand.friction_range[0], 
                            self._cfg.domain_rand.friction_range[1])

    @property
    def dr_restitution_values(self):
        """Returns the restitution values for domain randomization.

        Returns:
            Tensor((num_envs, 1)): Restitution values for domain randomization.
        """
        return dr_normalize(self._restitution_values, 
                            self._cfg.domain_rand.restitution_range[0], 
                            self._cfg.domain_rand.restitution_range[1])

    @property
    def dr_added_base_mass(self):
        """Returns the added base mass for domain randomization.
        
        Returns:
            Tensor((num_envs, 1)): Added base mass for domain randomization.
        """
        return dr_normalize(self._added_base_mass,
                            self._cfg.domain_rand.added_mass_range[0], 
                            self._cfg.domain_rand.added_mass_range[1])
    
    @property
    def dr_rand_push_vels(self):
        """Returns the random push velocities for domain randomization.

        Returns:
            Tensor((num_envs, 3)): Random push velocities for domain randomization.
        """
        return dr_normalize(self._rand_push_vels,
                            -self._cfg.domain_rand.max_push_vel_xy,
                            self._cfg.domain_rand.max_push_vel_xy)

    @property
    def dr_base_com_bias(self):
        """Returns the base COM bias for domain randomization.

        Returns:
            Tensor((num_envs, 3)): Base COM bias for domain randomization.
        """
        com_bias_x = dr_normalize(self._base_com_bias[:, 0],
                                  self._cfg.domain_rand.com_pos_x_range[0],
                                  self._cfg.domain_rand.com_pos_x_range[1])
        com_bias_y = dr_normalize(self._base_com_bias[:, 1],
                                  self._cfg.domain_rand.com_pos_y_range[0],
                                  self._cfg.domain_rand.com_pos_y_range[1])
        com_bias_z = dr_normalize(self._base_com_bias[:, 2],
                                  self._cfg.domain_rand.com_pos_z_range[0],
                                  self._cfg.domain_rand.com_pos_z_range[1])
        return torch.stack([com_bias_x, com_bias_y, com_bias_z], dim=-1)

    @property
    def dr_joint_armature(self):
        """Returns the joint armature for domain randomization.

        Returns:
            Tensor((num_envs, num_dof)): Joint armature for domain randomization.
        """
        return dr_normalize(self._joint_armature,
                            self._cfg.domain_rand.joint_armature_range[0],
                            self._cfg.domain_rand.joint_armature_range[1])
    
    @property
    def dr_joint_friction(self):
        """Returns the joint friction for domain randomization.

        Returns:
            Tensor((num_envs, num_dof)): Joint friction for domain randomization.
        """
        return dr_normalize(self._joint_friction,
                            self._cfg.domain_rand.joint_friction_range[0],
                            self._cfg.domain_rand.joint_friction_range[1])

    @property
    def dr_joint_damping(self):
        """Returns the joint damping for domain randomization.

        Returns:
            Tensor((num_envs, num_dof)): Joint damping for domain randomization.
        """
        return dr_normalize(self._joint_damping,
                            self._cfg.domain_rand.joint_damping_range[0],
                            self._cfg.domain_rand.joint_damping_range[1])

    @property
    def dr_kp_scale(self):
        """Returns the KP scale for domain randomization.

        Returns:
            Tensor((num_envs, num_dof)): KP scale for domain randomization.
        """
        return dr_normalize(self._kp_scale,
                            self._cfg.domain_rand.kp_range[0],
                            self._cfg.domain_rand.kp_range[1])

    @property
    def dr_kd_scale(self):
        """Returns the KD scale for domain randomization.

        Returns:
            Tensor((num_envs, num_dof)): KD scale for domain randomization.
        """
        return dr_normalize(self._kd_scale,
                            self._cfg.domain_rand.kd_range[0],
                            self._cfg.domain_rand.kd_range[1])

    @property
    def dr_ctrl_delay(self):
        """Returns the control delay for domain randomization.

        Returns:
            Tensor((num_envs, 1)): Control delay for domain randomization.
        """
        return dr_normalize(self._action_delay.unsqueeze(-1),
                            self._cfg.domain_rand.ctrl_delay_step_range[0],
                            self._cfg.domain_rand.ctrl_delay_step_range[1])
    
    @property
    def env_origins(self):
        """Returns the origin positions of all environments.

        Returns:
            Tensor((num_envs, 3)): Origin positions of all environments.
        """
        return self._env_origins
    
    @property
    def depth_images(self):
        """Returns the depth images from the depth camera.

        Returns:
            Tensor((num_envs, num_history, height, width)): Depth images from the depth camera.        
            The stack history is ordered from the latest(0) to the oldest(-1).
        """
        return self._depth_images