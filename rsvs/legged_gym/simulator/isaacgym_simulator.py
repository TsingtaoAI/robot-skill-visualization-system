from legged_gym import *
from legged_gym.simulator.simulator import Simulator
from PIL import Image as im
import torch
import numpy as np
import os
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
import warp as wp
import trimesh
from legged_gym.warp.warp_cam import WarpCam
if SIMULATOR == "isaacgym":
    from isaacgym import gymtorch, gymapi, gymutil

import warnings

""" ********** Isaac Gym Simulator ********** """
class IsaacGymSimulator(Simulator):
    def __init__(self, cfg : LeggedRobotCfg, sim_params: dict, sim_device: str = "cuda:0", headless: bool = False):
        self._gym = gymapi.acquire_gym()
        # Convert dict sim_params to gymapi.SimParams
        self._sim_params = gymapi.SimParams()
        gymutil.parse_sim_config(sim_params, self._sim_params)
        _, self._sim_device_id = gymutil.parse_device_str(sim_device)
        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self._sim_device_id
        if headless == True:
            self.graphics_device_id = -1
        self.physics_engine = gymapi.SIM_PHYSX
        super().__init__(cfg, sim_params, sim_device, headless)
        # warp init
        if self._cfg.sensor.add_depth:
            wp.init()
            self._create_warp_envs()
            self._create_warp_tensors()
            self._depth_camera_sensor = WarpCam(self._warp_tensor_dict, 
                                self._num_camera_envs, 
                                self._cfg.sensor, 
                                self._mesh_ids, 
                                self._device)
            pixels = self._depth_camera_sensor.update()
            self._depth_images[:,0] = pixels[:,0] # pixels: [num_envs, num_sensors, H, W]

    #----- Public methods -----#
    def step(self, actions):
        """Simulator steps, receiving actions from the agent"""
        self._render()
        self._last_base_lin_vel[:] = self._base_lin_vel[:]
        self._last_base_ang_vel[:] = self._base_ang_vel[:]
        self._last_feet_vel[:] = self._rigid_body_states[:, self._feet_indices, 7:10]
        actions = self._pre_simulator_step(actions)
        for _ in range(self._cfg.control.decimation):
            self._last_dof_vel[:] = self._dof_vel[:] # for computing velocity-target-based PD control
            self._torques = self._compute_torques(actions).view(self._torques.shape)
            self._gym.set_dof_actuation_force_tensor(self._sim, 
                                                gymtorch.unwrap_tensor(self._torques))
            self._gym.simulate(self._sim)
            if (self._device == "cpu"):
                self._gym.fetch_results(self._sim, True)
            self._gym.refresh_dof_state_tensor(self._sim)
            
    def post_physics_step(self):
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._check_base_pos_out_of_bound()
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        # the wrapped tensor will be updated automatically once you call refresh_xxx_tensor
        self._base_euler[:] = get_euler_xyz(self._base_quat)
        self._base_lin_vel[:] = quat_rotate_inverse(
            self._base_quat, self._root_states[:, 7:10])
        self._base_ang_vel[:] = quat_rotate_inverse(
            self._base_quat, self._root_states[:, 10:13])
        self._projected_gravity[:] = quat_rotate_inverse(
            self._base_quat, self._global_gravity)
        # Link contact state
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = 1. * (torch.norm(
                self._link_contact_forces[:, self._contact_state_link_indices, :], dim=-1) > 1.)
        # update terrain heights info
        if self._cfg.terrain.measure_heights:
            self._update_surrounding_heights()
            if self._cfg.terrain.obtain_terrain_info_around_feet:
                self._calc_terrain_info_around_feet()
        # Refresh warp sensor pose
        if self._cfg.sensor.add_depth:
            sensor_quat = quat_mul(self._base_quat[:self._num_camera_envs], self._sensor_offset_quat)
            sensor_pos = self._base_pos[:self._num_camera_envs] + quat_apply(self._base_quat[:self._num_camera_envs], self._sensor_offset_pos)
            self._sensor_pos_tensor[:,:] = sensor_pos[:,:]
            self._sensor_quat_tensor[:,:] = sensor_quat[:,:]
    
    def reset_idx(self, env_ids):
        # rigid body props and joint props in IsaacGym can not be modified on the fly
        # resample pd randomization params
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(env_ids)
        
        self._last_dof_vel[env_ids] = 0.
        self._last_feet_vel[env_ids] = 0.
        self._last_base_lin_vel[env_ids] = 0.
        self._last_base_ang_vel[env_ids] = 0.
        
        # reset action queue and delay
        if self._cfg.domain_rand.randomize_ctrl_delay:
            self._action_queue[env_ids] *= 0.
            self._action_queue[env_ids] = 0.
            self._action_delay[env_ids] = torch.randint(self._cfg.domain_rand.ctrl_delay_step_range[0],
                                                       self._cfg.domain_rand.ctrl_delay_step_range[1]+1, (len(env_ids),), device=self._device, requires_grad=False)
        
        # reset depth image tensors
        # find common ids between env_ids and camera env ids
        if self._cfg.sensor.add_depth:
            camera_env_ids = torch.arange(self._num_camera_envs, device=self._device)
            common_env_ids = torch.tensor(
                [env_id for env_id in env_ids.tolist() if env_id in camera_env_ids.tolist()],
                device=self._device, dtype=torch.long)
            self._depth_images[common_env_ids, :] = 0.
            
    def reset_dofs(self, env_ids, dof_pos, dof_vel):
        self._dof_pos[env_ids] = dof_pos[:, self._dof_indices]
        self._dof_vel[env_ids] = dof_vel[:, self._dof_indices]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self._gym.set_dof_state_tensor_indexed(self._sim,
                                               gymtorch.unwrap_tensor(self._dof_state),
                                               gymtorch.unwrap_tensor(env_ids_int32), 
                                               len(env_ids_int32))
        
    def reset_root_states(self, 
                          env_ids, 
                          base_pos, 
                          base_quat, 
                          base_lin_vel_w, 
                          base_ang_vel_w):
        # base position
        self._root_states[env_ids, 0:3] = base_pos[:]
        # base orientation
        self._root_states[env_ids, 3:7] = base_quat[:]
        # base velocities in world frame
        self._root_states[env_ids, 7:10] = base_lin_vel_w[:]
        self._root_states[env_ids, 10:13] = base_ang_vel_w[:]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self._gym.set_actor_root_state_tensor_indexed(self._sim,
                                                      gymtorch.unwrap_tensor(self._root_states),
                                                      gymtorch.unwrap_tensor(env_ids_int32), 
                                                      len(env_ids_int32))
        # update data buffers
        self._base_euler[env_ids] = get_euler_xyz(self._base_quat[env_ids])
        self._projected_gravity[env_ids] = quat_rotate_inverse(
            self._base_quat[env_ids], self._global_gravity[env_ids])
        self._base_lin_vel[env_ids] = quat_rotate_inverse(
            self._base_quat[env_ids], self._root_states[env_ids, 7:10])
        self._base_ang_vel[env_ids] = quat_rotate_inverse(
            self._base_quat[env_ids], self._root_states[env_ids, 10:13])
        self._gym.refresh_rigid_body_state_tensor(self._sim)
    
    def update_terrain_curriculum(self, env_ids, move_up, move_down):
        self._terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self._terrain_levels[env_ids] = torch.where(self._terrain_levels[env_ids] >=self._max_terrain_level,
                                                   torch.randint_like(
                                                       self._terrain_levels[env_ids], self._max_terrain_level),
                                                   torch.clip(self._terrain_levels[env_ids], 0))  # (the minumum level is zero)
        self._env_origins[env_ids] = self._terrain_origins[self._terrain_levels[env_ids],
            self._terrain_types[env_ids]]
    
    def push_robots(self):
        max_vel = self._cfg.domain_rand.max_push_vel_xy
        self._rand_push_vels[:, :2] = torch_rand_float(-max_vel, max_vel, (self._num_envs, 2), device=self._device)
        self._root_states[:, 7:9] += self._rand_push_vels[:, :2] # set random base velocity in xy plane
        self._gym.set_actor_root_state_tensor(self._sim, gymtorch.unwrap_tensor(self._root_states))
        self._last_base_lin_vel[:] = self._base_lin_vel[:]
        self._base_lin_vel[:] = quat_rotate_inverse(
            self._base_quat, self._root_states[:, 7:10])
    
    def push_links(self):
        max_force = self._cfg.domain_rand.max_push_force
        push_force = torch.rand((self._num_envs, self._num_bodies, 3), 
                                device=self._device) * 2 * max_force - max_force
        self._gym.apply_rigid_body_force_tensors(
            self._sim,
            gymtorch.unwrap_tensor(push_force),
            None,
            gymapi.CoordinateSpace.GLOBAL_SPACE
        )
    
    def draw_debug_vis(self, ref_key_body_pos=None):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        # if not self._cfg.terrain.measure_heights:
        #     return
        self._gym.clear_lines(self._viewer)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        if self._cfg.env.debug_draw_height_points_around_base:
            self._draw_height_points_around_base()
        if self._cfg.env.debug_draw_height_points_around_feet:
            self._draw_height_points_around_feet()
        if self._cfg.env.debug_draw_terrain_height_points:
            self._draw_terrain_height_points()
        if self._cfg.env.debug_draw_terrain_edge_points:
            self._draw_terrain_edge_points()
        if self._cfg.env.debug_draw_key_body_points:
            self._draw_key_body_points(ref_key_body_pos)
        if self._cfg.env.debug_draw_depth_images:
            self._draw_debug_depth_images()
    
    def set_viewer_camera(self, eye: np.ndarray, target: np.ndarray):
        cam_pos = gymapi.Vec3(eye[0], eye[1], eye[2])
        cam_target = gymapi.Vec3(target[0], target[1], target[2])
        self._gym.viewer_camera_look_at(self._viewer, None, cam_pos, cam_target)
    
    def update_sensors(self):
        """ Update depth images from the depth camera sensors
        """
        if self._cfg.sensor.add_depth:
            if self._depth_image_update_counter % self._depth_image_update_decimation == 0:
                self._update_depth_camera()
            self._depth_image_update_counter += 1
    
    def get_height_at(self, pos_xy):
        """ Get height of the terrain at specific (x, y) location

        Args:
            pos_xy (torch.tensor): (x, y) locations to sample heights at, shape: (len(env_ids), 2)
        Returns:
            torch.tensor: heights at the specified (x, y) locations, shape: (len(env_ids),)
        """
        if self._cfg.terrain.mesh_type == 'plane':
            return torch.zeros_like(pos_xy[:, 0], device=self._device, requires_grad=False)
        elif self._cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        points = pos_xy + self._cfg.terrain.border_size # add border size to align the origin with heightfield raw
        points = (points/self._cfg.terrain.horizontal_scale).long()
        px = points[:, 0]
        py = points[:, 1]
        px = torch.clip(px, 0, self._height_samples.shape[0]-2)
        py = torch.clip(py, 0, self._height_samples.shape[1]-2)
        
        return self._height_samples[px, py] * self._cfg.terrain.vertical_scale

    def calc_feet_near_edge(self):
        """ Calculate whether each foot is near the terrain edge, which can be used as a termination condition or for reward shaping
        Returns:
            torch.tensor: whether each foot is near the terrain edge, shape: (num_envs, num_feet)
        """
        if self._cfg.terrain.mesh_type == 'plane':
            return torch.zeros((self._num_envs, len(self._feet_indices)), device=self._device, dtype=torch.bool)
        elif self._cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't calculate feet near edge with terrain mesh type 'none'")
        
        feet_pos_xy = self._rigid_body_states[:, self._feet_indices, :2] # (num_envs, num_feet, 2)
        feet_points_float = feet_pos_xy + self._cfg.terrain.border_size # add border size to align the origin with heightfield raw
        points = (feet_points_float/self._cfg.terrain.horizontal_scale).long()
        px = points[:,:, 0]
        py = points[:,:, 1]
        px = torch.clip(px, 0, self._edge_mask.shape[0]-1)
        py = torch.clip(py, 0, self._edge_mask.shape[1]-1)
        
        # if any of the four points around the foot is an edge point
        # and the position of the foot is within 5 cm to the edge point, we consider the foot to be near the edge
        edge_threshold = self._cfg.rewards.feet_edge_threshold
        foot_near_edge = torch.zeros((self._num_envs, len(self._feet_indices)), device=self._device, dtype=torch.bool)
        for i in range(-1, 2):
            for j in range(-1, 2):
                edge_points = self._edge_mask[torch.clip(px+i, 0, self._edge_mask.shape[0]-1), torch.clip(py+j, 0, self._edge_mask.shape[1]-1)] == 1
                if edge_points.any():
                    edge_points_pos = torch.zeros((self._num_envs, len(self._feet_indices), 2), device=self._device)
                    edge_points_pos[:,:,0] = (torch.clip(px+i, 0, self._edge_mask.shape[0]-1).float() * self._cfg.terrain.horizontal_scale) - self._cfg.terrain.border_size
                    edge_points_pos[:,:,1] = (torch.clip(py+j, 0, self._edge_mask.shape[1]-1).float() * self._cfg.terrain.horizontal_scale) - self._cfg.terrain.border_size
                    foot_near_edge = foot_near_edge | (edge_points & (torch.norm(feet_pos_xy - edge_points_pos, dim=-1) < edge_threshold))
        
        return foot_near_edge
    
    #----- Protected methods -----#
    def _pre_simulator_step(self, actions):
        # apply action delay by using an action queue
        if self._cfg.domain_rand.randomize_ctrl_delay:
            self._action_queue[:, 1:] = self._action_queue[:, :-1].clone()
            self._action_queue[:, 0] = actions.clone()
            actions = self._action_queue[torch.arange(
                self._num_envs), self._action_delay].clone()
        
        return actions
    
    def _parse_cfg(self):
        self._debug = self._cfg.env.debug
        self._control_dt = self._cfg.sim.dt * self._cfg.control.decimation
        if self._cfg.sensor.add_depth:
            self._num_camera_envs = self._cfg.env.num_camera_envs
            # update counter for depth images
            self._depth_image_update_counter = 0
            self._depth_image_update_decimation = self._cfg.sensor.depth_camera_config.decimation
    
    def _create_sim(self):
        self._up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self._sim = self._gym.create_sim(self._sim_device_id, self.graphics_device_id, self.physics_engine, self._sim_params)
        mesh_type = self._cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self._terrain = Terrain(self._cfg.terrain)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        
        # specify the boundary of the heightfield
        self._terrain_x_range = torch.zeros(2, device=self._device)
        self._terrain_y_range = torch.zeros(2, device=self._device)
        if self._cfg.terrain.mesh_type == 'heightfield' or self._cfg.terrain.mesh_type == 'trimesh':
            # give a small margin(1.0m)
            self._terrain_x_range[0] = -self._cfg.terrain.border_size + 1.0
            self._terrain_x_range[1] = self._cfg.terrain.border_size + \
                self._cfg.terrain.num_rows * self._cfg.terrain.terrain_length - 1.0
            self._terrain_y_range[0] = -self._cfg.terrain.border_size + 1.0
            self._terrain_y_range[1] = self._cfg.terrain.border_size + \
                self._cfg.terrain.num_cols * self._cfg.terrain.terrain_width - 1.0
        elif self._cfg.terrain.mesh_type == 'plane':  # the plane used has limited size,
            pass
        
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self._cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self._cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self._cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self._cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self._cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self._cfg.asset.fix_base_link
        asset_options.density = self._cfg.asset.density
        asset_options.angular_damping = self._cfg.asset.angular_damping
        asset_options.linear_damping = self._cfg.asset.linear_damping
        asset_options.max_angular_velocity = self._cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self._cfg.asset.max_linear_velocity
        asset_options.armature = self._cfg.asset.armature
        asset_options.thickness = self._cfg.asset.thickness
        asset_options.disable_gravity = self._cfg.asset.disable_gravity

        robot_asset = self._gym.load_asset(self._sim, asset_root, asset_file, asset_options)
        dof_props_asset = self._gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self._gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        self._body_names = self._gym.get_asset_rigid_body_names(robot_asset)
        print(f"body_names: {self._body_names}")
        self._dof_names = self._gym.get_asset_dof_names(robot_asset)
        print(f"dof_names: {self._dof_names}")
        self._dof_indices = [self._dof_names.index(name) for name in self._cfg.asset.dof_names]
        print(f"dof_indices: {self._dof_indices}")
        self._num_bodies = len(self._body_names)
        self._num_dof = len(self._dof_names)
        self._feet_names = [s for s in self._body_names if self._cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self._cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self._body_names if name in s])
        termination_contact_names = []
        for name in self._cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self._body_names if name in s])
        key_body_names = []
        for name in self._cfg.asset.key_bodies:
            key_body_names.extend([s for s in self._body_names if name in s])
        if self._cfg.asset.obtain_link_contact_states:
            contact_state_link_names = []
            for name in self._cfg.asset.contact_state_link_names:
                contact_state_link_names.extend([s for s in self._body_names if name in s])

        self._base_init_pos = torch.tensor(self._cfg.init_state.pos, dtype=torch.float, device=self._device, requires_grad=False)
        self._base_init_quat = torch.tensor(self._cfg.init_state.rot, dtype=torch.float, device=self._device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self._base_init_pos)

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self._actor_handles = []
        self._envs = []
        
        # privileged information
        self._init_domain_params()
        
        for i in range(self._num_envs):
            # create env instance
            env_handle = self._gym.create_env(self._sim, env_lower, env_upper, int(np.sqrt(self._num_envs)))
            pos = self._env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self._device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self._gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self._gym.create_actor(env_handle, robot_asset, start_pose, self._cfg.asset.name, i, self._cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self._gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self._gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self._gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self._envs.append(env_handle)
            self._actor_handles.append(actor_handle)

        self._feet_indices = torch.zeros(len(self._feet_names), dtype=torch.long, device=self._device, requires_grad=False)
        for i in range(len(self._feet_names)):
            self._feet_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actor_handles[0], self._feet_names[i])

        self._penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self._device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self._penalized_contact_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actor_handles[0], penalized_contact_names[i])

        self._termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self._device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self._termination_contact_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actor_handles[0], termination_contact_names[i])
        
        self._base_link_index = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actor_handles[0], self._cfg.asset.base_link_name)
        print(f"Base link index: {self._base_link_index}")
        
        self._key_body_indices = torch.zeros(len(key_body_names), dtype=torch.long, device=self._device, requires_grad=False)
        for i in range(len(key_body_names)):
            self._key_body_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actor_handles[0], key_body_names[i])
        print(f"Key body names: {key_body_names}")
        print(f"Key body indices: {self._key_body_indices}")
        
        if self._cfg.asset.obtain_link_contact_states:
            self._contact_state_link_indices = torch.zeros(len(contact_state_link_names), dtype=torch.long, device=self._device, requires_grad=False)
            for i in range(len(contact_state_link_names)):
                self._contact_state_link_indices[i] = self._gym.find_actor_rigid_body_handle(self._envs[0], self._actor_handles[0], contact_state_link_names[i])

        self._gym.prepare_sim(self._sim)
        # todo: read from config
        self._enable_viewer_sync = True
        self._viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self._headless == False:
            # subscribe to keyboard shortcuts
            self._viewer = self._gym.create_viewer(
                self._sim, gymapi.CameraProperties())
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_ESCAPE, "QUIT")
            self._gym.subscribe_viewer_keyboard_event(
                self._viewer, gymapi.KEY_V, "toggle_viewer_sync")
    
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self._gym.acquire_actor_root_state_tensor(self._sim)
        dof_state_tensor = self._gym.acquire_dof_state_tensor(self._sim)
        net_contact_forces = self._gym.acquire_net_contact_force_tensor(self._sim)
        rigid_body_state = self._gym.acquire_rigid_body_state_tensor(self._sim)
        self._gym.refresh_dof_state_tensor(self._sim)
        self._gym.refresh_actor_root_state_tensor(self._sim)
        self._gym.refresh_net_contact_force_tensor(self._sim)
        self._gym.refresh_rigid_body_state_tensor(self._sim)

        # create some wrapper tensors for different slices
        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        self._rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self._num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self._num_envs, self._num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self._num_envs, self._num_dof, 2)[..., 1]
        self._base_pos = self._root_states[:, 0:3]
        self._base_quat = self._root_states[:, 3:7]
        self._last_dof_pos = torch.zeros_like(self._dof_pos)
        self._base_euler = get_euler_xyz(self._base_quat)
        self._link_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self._num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self._last_feet_vel = torch.zeros_like(self._rigid_body_states[:, self._feet_indices, 7:10])

        # initialize some data used later on
        self._global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self._device, dtype=torch.float).repeat(self._num_envs, 1)
        self._torques = torch.zeros(self._num_envs, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self._p_gains = torch.zeros(self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self._d_gains = torch.zeros(self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self._last_dof_vel = torch.zeros_like(self._dof_vel)
        self._base_lin_vel = quat_rotate_inverse(self._base_quat, self._root_states[:, 7:10])
        self._base_ang_vel = quat_rotate_inverse(self._base_quat, self._root_states[:, 10:13])
        self._last_base_lin_vel = torch.zeros_like(self._base_lin_vel)
        self._last_base_ang_vel = torch.zeros_like(self._base_ang_vel)
        self._projected_gravity = quat_rotate_inverse(self._base_quat, self._global_gravity)
        
        # Terrain information around feet
        if self._cfg.terrain.obtain_terrain_info_around_feet:
            self._normal_vector_around_feet = torch.zeros(
                self._num_envs, len(self._feet_indices) * 3, dtype=torch.float, device=self._device, requires_grad=False)
            self._height_around_feet = torch.zeros(
                self._num_envs, len(self._feet_indices), 9, dtype=torch.float, device=self._device, requires_grad=False)
        
        # Link contact state
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = torch.zeros(
                self._num_envs, len(self._contact_state_link_indices), dtype=torch.float, device=self._device, requires_grad=False)

        # joint positions offsets and PD gains
        self._default_dof_pos = torch.zeros(self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)
        for i in range(self._num_dof):
            # self._p_gains use the joint sequence in simulation
            name = self._dof_names[i]
            # default_dof_pos use the joint sequence in simulation
            self._default_dof_pos[i] = self._cfg.init_state.default_joint_angles[name]
            found = False
            for dof_name in self._cfg.control.stiffness.keys():
                if dof_name in name:
                    self._p_gains[i] = self._cfg.control.stiffness[dof_name]
                    self._d_gains[i] = self._cfg.control.damping[dof_name]
                    found = True
            if not found:
                self._p_gains[i] = 0.
                self._d_gains[i] = 0.
                if self._cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self._default_dof_pos = self._default_dof_pos.unsqueeze(0)
        
        # control delay
        if self._cfg.domain_rand.randomize_ctrl_delay:
            self._action_queue = torch.zeros(
                self._num_envs, self._cfg.domain_rand.ctrl_delay_step_range[1]+1, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
            self._action_delay = torch.randint(self._cfg.domain_rand.ctrl_delay_step_range[0],
                                              self._cfg.domain_rand.ctrl_delay_step_range[1]+1, (self._num_envs,), device=self._device, requires_grad=False)
        
        # Camera image tensor
        if self._cfg.sensor.add_depth:
            pointcloud_dims = 3 * (self._cfg.sensor.depth_camera_config.return_pointcloud == True)
            if pointcloud_dims > 0: # pointcloud returned by depth camera
                self._depth_images = torch.zeros(
                    (self._num_camera_envs, 
                    self._cfg.sensor.depth_camera_config.num_history,
                    *self._cfg.sensor.depth_camera_config.resolution,
                    pointcloud_dims), 
                    device=self._device, 
                    dtype=torch.float
                )
            else:
                self._depth_images = torch.zeros(
                    (self._num_camera_envs, 
                    self._cfg.sensor.depth_camera_config.num_history,
                    *self._cfg.sensor.depth_camera_config.resolution), 
                    device=self._device, 
                    dtype=torch.float
                )
        
        self._init_height_points()
        self._measured_heights = torch.zeros(self._num_envs, self._num_height_points, device=self._device, requires_grad=False)
    
    def _init_height_points(self):
        y = torch.tensor(self._cfg.terrain.measured_points_y,
                         device=self._device, requires_grad=False)
        x = torch.tensor(self._cfg.terrain.measured_points_x,
                         device=self._device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self._num_height_points = grid_x.numel()
        self._height_points = torch.zeros(self._num_envs, self._num_height_points,
                             3, device=self._device, requires_grad=False)
        self._height_points[:, :, 0] = grid_x.flatten()
        self._height_points[:, :, 1] = grid_y.flatten()
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self._cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self._custom_origins = True
            self._env_origins = torch.zeros(self._num_envs, 3, device=self._device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self._cfg.terrain.max_init_terrain_level
            if not self._cfg.terrain.curriculum: max_init_level = self._cfg.terrain.num_rows - 1
            self._terrain_levels = torch.randint(0, max_init_level+1, (self._num_envs,), device=self._device)
            self._terrain_types = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
            if hasattr(self._cfg.env, "num_camera_envs"): # split envs with cameras to all terrain types, ensuring the diversity of data collected by cameras
                self._terrain_types[:self._cfg.env.num_camera_envs] = torch.div(
                    torch.arange(self._cfg.env.num_camera_envs, device=self._device), (self._cfg.env.num_camera_envs/self._cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
                self._terrain_types[self._cfg.env.num_camera_envs:] = torch.div(
                    torch.arange(self._num_envs - self._cfg.env.num_camera_envs, device=self._device), ((self._num_envs - self._cfg.env.num_camera_envs)/self._cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            else:
                self._terrain_types = torch.div(torch.arange(self._num_envs, device=self._device), (self._num_envs/self._cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self._max_terrain_level = self._cfg.terrain.num_rows
            self._terrain_origins = torch.from_numpy(self._terrain.env_origins).to(self._device).to(torch.float)
            self._env_origins[:] = self._terrain_origins[self._terrain_levels, self._terrain_types]
        else:
            self._custom_origins = False
            self._env_origins = torch.zeros(self._num_envs, 3, device=self._device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self._num_envs))
            num_rows = np.ceil(self._num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self._cfg.env.env_spacing
            self._env_origins[:, 0] = spacing * xx.flatten()[:self._num_envs]
            self._env_origins[:, 1] = spacing * yy.flatten()[:self._num_envs]
            self._env_origins[:, 2] = 0.
    
    def _update_surrounding_heights(self):
        if self._cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self._num_envs, self._num_height_points, device=self._device, requires_grad=False)
        elif self._cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        points = quat_apply_yaw(self._base_quat.repeat(
                1, self._num_height_points), self._height_points) + (self._base_pos[:, :3]).unsqueeze(1)

        # When acquiring heights, the points need to add border_size
        # because in the height_samples, the origin of the terrain is at (border_size, border_size)
        points += self._cfg.terrain.border_size
        points = (points/self._cfg.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self._height_samples.shape[0]-2)
        py = torch.clip(py, 0, self._height_samples.shape[1]-2)

        heights1 = self._height_samples[px, py]
        heights2 = self._height_samples[px+1, py]
        heights3 = self._height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        self._measured_heights = heights.view(self._num_envs, -1) * self._cfg.terrain.vertical_scale
    
    def _calc_terrain_info_around_feet(self):
        # Foot position
        feet_pos = self._rigid_body_states[:, self._feet_indices, 0:3]
        foot_points = feet_pos + self._cfg.terrain.border_size
        foot_points = (foot_points/self._cfg.terrain.horizontal_scale).long()
        # px and py for 4 feet, num_envs*len(feet_indices)
        px = foot_points[:, :, 0].view(-1)
        py = foot_points[:, :, 1].view(-1)
        # clip to the range of height samples
        px = torch.clip(px, 0, self._height_samples.shape[0]-2)
        py = torch.clip(py, 0, self._height_samples.shape[1]-2)
        # get heights around the feet, 9 points for each foot
        heights1 = self._height_samples[px-1, py]  # [x-0.1, y]
        heights2 = self._height_samples[px+1, py]  # [x+0.1, y]
        heights3 = self._height_samples[px, py-1]  # [x, y-0.1]
        heights4 = self._height_samples[px, py+1]  # [x, y+0.1]
        heights5 = self._height_samples[px, py]    # [x, y]
        heights6 = self._height_samples[px-1, py-1]  # [x-0.1, y-0.1]
        heights7 = self._height_samples[px+1, py+1]  # [x+0.1, y+0.1]
        heights8 = self._height_samples[px-1, py+1]  # [x-0.1, y+0.1]
        heights9 = self._height_samples[px+1, py-1]  # [x+0.1, y-0.1]
        # Calculate normal vectors around feet
        dx = ((heights2 - heights1) / (self._cfg.terrain.horizontal_scale * 2)).view(self._num_envs, -1)
        dy = ((heights4 - heights3) / (self._cfg.terrain.horizontal_scale * 2)).view(self._num_envs, -1)
        for i in range(len(self._feet_indices)):
            normal_vector = torch.cat((dx[:, i].unsqueeze(1), dy[:, i].unsqueeze(1), 
                -1*torch.ones_like(dx[:, i].unsqueeze(1))), dim=-1).to(self._device)
            normal_vector /= torch.norm(normal_vector, dim=-1, keepdim=True)
            self._normal_vector_around_feet[:, i*3:i*3+3] = normal_vector[:]
        # Calculate height around feet
        for i in range(9):
            self._height_around_feet[:, :, i] = eval(f'heights{i+1}').view(self._num_envs, -1)[:] * self._cfg.terrain.vertical_scale
    
    def _check_base_pos_out_of_bound(self):
        """ Check if the base position is out of the terrain bounds
        """
        if self._cfg.terrain.mesh_type == "plane":
            return
        x_out_of_bound = (self._base_pos[:, 0] >= self._terrain_x_range[1]) | (
            self._base_pos[:, 0] <= self._terrain_x_range[0])
        y_out_of_bound = (self._base_pos[:, 1] >= self._terrain_y_range[1]) | (
            self._base_pos[:, 1] <= self._terrain_y_range[0])
        out_of_bound_buf = x_out_of_bound | y_out_of_bound
        env_ids = out_of_bound_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        else:
            # reset base position to initial position
            self._root_states[env_ids, 0:3] = self._base_init_pos
            self._root_states[env_ids, 0:3] += self._env_origins[env_ids]
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self._gym.set_actor_root_state_tensor_indexed(self._sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), 
                                                     len(env_ids_int32))
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self._cfg.control.action_scale
        control_type = self._cfg.control.control_type
        # convert actions from policy joint sequence to simulation joint sequence
        if control_type=="P":
            torques = self._kp_scale * self._p_gains * (actions_scaled[:, self._dof_indices] + \
                self._default_dof_pos - self._dof_pos) - \
                    self._kd_scale * self._d_gains * self._dof_vel
        elif control_type=="V":
            torques = self._kp_scale * self._p_gains * (actions_scaled[:, self._dof_indices] - \
                self._dof_vel) - self._kd_scale * self._d_gains * \
                    (self._dof_vel - self._last_dof_vel)/self._sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self._torque_limits, self._torque_limits)
    
    def _init_domain_params(self):
        """ Initializes domain randomization parameters, which are used to randomize the environment."""
        self._friction_values = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._restitution_values = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._added_base_mass = torch.ones(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._rand_push_vels = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self._base_com_bias = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_armature = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_friction = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_damping = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._kp_scale = torch.ones(
            self._num_envs, self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)
        self._kd_scale = torch.ones(
            self._num_envs, self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)

    def _randomize_friction(self, env_ids):
        return super()._randomize_friction(env_ids)

    def _randomize_restitution(self, env_ids):
        return super()._randomize_restitution(env_ids)

    def _randomize_base_mass(self, env_ids):
        return super()._randomize_base_mass(env_ids)
    
    def _randomize_com_displacement(self, env_ids):
        return super()._randomize_com_displacement(env_ids)
    
    def _randomize_joint_armature(self, env_ids):
        return super()._randomize_joint_armature(env_ids)
    
    def _randomize_joint_friction(self, env_ids):
        return super()._randomize_joint_friction(env_ids)
    
    def _randomize_joint_damping(self, env_ids):
        return super()._randomize_joint_damping(env_ids)
    
    def _randomize_pd_gain(self, env_ids):
        self._kp_scale[env_ids] = torch_rand_float(
                self._cfg.domain_rand.kp_range[0], self._cfg.domain_rand.kp_range[1], (len(env_ids), self._num_actions), device=self._device)
        self._kd_scale[env_ids] = torch_rand_float(
                self._cfg.domain_rand.kd_range[0], self._cfg.domain_rand.kd_range[1], (len(env_ids), self._num_actions), device=self._device)
    
    def _update_depth_camera(self):
        near_clip = self._cfg.sensor.depth_camera_config.near_clip
        far_clip = self._cfg.sensor.depth_camera_config.far_clip
        pixels = self._depth_camera_sensor.update().clone()
        if self._depth_images.shape[1] > 1: # stack history of depth images
            self._depth_images[:, 1:] = self._depth_images[:, :-1].detach().clone()
        # store values for denoised depth images
        self._depth_images[:, 0] = pixels[:,0,:,:] # pixels: [num_envs, num_sensors, H, W]
        # clip values
        self._depth_images[:, 0] = torch.clip(self._depth_images[:, 0], near_clip, far_clip)
        # normalize the depth images to be within [-0.5, 0.5]
        self._depth_images[:, 0] = (self._depth_images[:, 0] - near_clip) / (far_clip - near_clip) - 0.5
    
    def _create_warp_envs(self):
        terrain_mesh = self._terrain.terrain_mesh
        
        #save terrain mesh
        transform = np.zeros((3,))
        transform[0] = -self._cfg.terrain.border_size 
        transform[1] = -self._cfg.terrain.border_size
        transform[2] = 0.0
        translation = trimesh.transformations.translation_matrix(transform)
        terrain_mesh.apply_transform(translation)
        
        vertices = terrain_mesh.vertices
        triangles = terrain_mesh.faces
        vertex_tensor = torch.tensor(vertices, 
                                     dtype=torch.float32, 
                                     device=self._device,
                                     requires_grad=False)
        #if none type in vertex_tensor
        if vertex_tensor.any() is None:
            print("vertex_tensor is None")
        vertex_vec3_array = wp.from_torch(vertex_tensor,dtype=wp.vec3)        
        faces_wp_int32_array = wp.from_numpy(triangles.flatten(), dtype=wp.int32,device=self._device)
        
        # mesh coordinate convention may be the same as isaacgym
        self._wp_meshes =  wp.Mesh(points=vertex_vec3_array,indices=faces_wp_int32_array)
        
        Warning("Currently, only static terrain mesh is added to Warp.")
        self._mesh_ids = self.mesh_ids_array = wp.array([self._wp_meshes.id], dtype=wp.uint64)
    
    def _create_warp_tensors(self):
        self._warp_tensor_dict={}
        pointcloud_dims = 3 * (self._cfg.sensor.depth_camera_config.return_pointcloud == True)
        if pointcloud_dims > 0:
            self._depth_image_tensor_warp = torch.zeros((self._num_camera_envs, 
                                                        self._cfg.sensor.depth_camera_config.num_sensors,
                                                        *self._cfg.sensor.depth_camera_config.resolution,
                                                        pointcloud_dims),    # xyz
                                                       dtype=torch.float32, device=self._device)
        else:
            self._depth_image_tensor_warp = torch.zeros((self._num_camera_envs, 
                                                        self._cfg.sensor.depth_camera_config.num_sensors,
                                                        *self._cfg.sensor.depth_camera_config.resolution),
                                                    dtype=torch.float32, device=self._device)
        self._sensor_pos_tensor = torch.zeros_like(self._root_states[:self._num_camera_envs, 0:3])
        self._sensor_quat_tensor = torch.zeros_like(self._root_states[:self._num_camera_envs, 3:7])

        # sensor pose
        pos_offset = [self._cfg.sensor.depth_camera_config.pos[0], 
                      self._cfg.sensor.depth_camera_config.pos[1], 
                      self._cfg.sensor.depth_camera_config.pos[2]]
        rpy_offset = [self._cfg.sensor.depth_camera_config.euler[0], 
                      self._cfg.sensor.depth_camera_config.euler[1], 
                      self._cfg.sensor.depth_camera_config.euler[2]]
        self._sensor_offset_pos = torch.tensor(pos_offset, device=self._device).repeat((self._num_camera_envs, 1))
        rpy_offset = torch.tensor(rpy_offset, device=self._device).repeat((self._num_camera_envs, 1))
        
        # apply domain randomization to sensor position and sensor rotation
        if self._cfg.domain_rand.randomize_camera_pos:
          self._camera_pos_offset = torch.zeros(
                self._num_camera_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
          self._camera_pos_offset[:self._num_camera_envs, 0] = torch_rand_float(
                -self._cfg.domain_rand.camera_com_displacement_range[0],
                self._cfg.domain_rand.camera_com_displacement_range[0],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          self._camera_pos_offset[:self._num_camera_envs, 1] = torch_rand_float(
                -self._cfg.domain_rand.camera_com_displacement_range[1],
                self._cfg.domain_rand.camera_com_displacement_range[1],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          self._camera_pos_offset[:self._num_camera_envs, 2] = torch_rand_float(
                -self._cfg.domain_rand.camera_com_displacement_range[2],
                self._cfg.domain_rand.camera_com_displacement_range[2],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          self._sensor_offset_pos += self._camera_pos_offset[:self._num_camera_envs]
          
        if self._cfg.domain_rand.randomize_camera_euler:
          self._camera_euler_offset = torch.zeros(
                self._num_camera_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
          self._camera_euler_offset[:self._num_camera_envs, 0] = torch_rand_float(
                -self._cfg.domain_rand.camera_euler_range[0],
                self._cfg.domain_rand.camera_euler_range[0],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          self._camera_euler_offset[:self._num_camera_envs, 1] = torch_rand_float(
                -self._cfg.domain_rand.camera_euler_range[1],
                self._cfg.domain_rand.camera_euler_range[1],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          self._camera_euler_offset[:self._num_camera_envs, 2] = torch_rand_float(
                -self._cfg.domain_rand.camera_euler_range[2],
                self._cfg.domain_rand.camera_euler_range[2],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          rpy_offset += self._camera_euler_offset[:self._num_camera_envs]
          
        self._sensor_offset_quat = quat_from_euler_xyz(rpy_offset[:, 0], 
                                                       rpy_offset[:, 1], 
                                                       rpy_offset[:, 2])
        
        self._warp_tensor_dict["depth_image_tensor"] = self._depth_image_tensor_warp
        self._warp_tensor_dict['device'] = self._device
        self._warp_tensor_dict['num_envs'] = self._num_camera_envs
        self._warp_tensor_dict['num_sensors'] = self._cfg.sensor.depth_camera_config.num_sensors
        self._warp_tensor_dict['sensor_pos_tensor'] = self._sensor_pos_tensor
        self._warp_tensor_dict['sensor_quat_tensor'] = self._sensor_quat_tensor
        self._warp_tensor_dict['mesh_ids'] = self._mesh_ids
    
    def _draw_debug_points_world(self, points, radius=0.02, color=(1, 0, 0)):
        """ Draws debug points in world frame

        Args:
            points (torch.tensor): points to draw, shape: (num_envs, num_points, 3)
            color (tuple, optional): RGB color of the points. Defaults to (1, 0, 0).
        """
        sphere_geom = gymutil.WireframeSphereGeometry(radius, 4, 4, None, color=color)
        for i in range(self._num_envs):
            for j in range(points.shape[1]):
                x = points[i, j, 0].cpu().numpy()
                y = points[i, j, 1].cpu().numpy()
                z = points[i, j, 2].cpu().numpy()
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self._gym, self._viewer, self._envs[i], sphere_pose)
    
    
    def _draw_terrain_height_points(self):
        """ Draws height measurement points in the terrain for debugging
        """
        # draw height lines
        if not self._cfg.terrain.measure_heights:
            return
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 8, 8, None, color=(1, 1, 0))
        for i in range(self._height_samples.shape[0]):
            for j in range(self._height_samples.shape[1]):
                x = i * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
                y = j * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
                z = self._height_samples[i, j].cpu().numpy() * self._cfg.terrain.vertical_scale
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self._gym, self._viewer, self._envs[0], sphere_pose)
    
    def _draw_terrain_edge_points(self):
        """ Draws terrain edge points for debugging
        """
        if not self._cfg.terrain.measure_heights:
            return
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 8, 8, None, color=(1, 0, 1))
        for i in range(self._edge_mask.shape[0]):
            for j in range(self._edge_mask.shape[1]):
                if self._edge_mask[i, j] == 1:
                    x = i * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
                    y = j * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
                    z = self._height_samples[i, j].cpu().numpy() * self._cfg.terrain.vertical_scale
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self._gym, self._viewer, self._envs[0], sphere_pose)
    
    def _draw_key_body_points(self, ref_key_body_pos=None):
        """ Draws key body points for debugging
        """
        if ref_key_body_pos is not None:
            sphere_geom = gymutil.WireframeSphereGeometry(0.03, 8, 8, None, color=(0, 1, 0))
            for i in range(self._num_envs):
                for j in range(self._key_body_indices.shape[0]):
                    x = ref_key_body_pos[i, j, 0].cpu().numpy()
                    y = ref_key_body_pos[i, j, 1].cpu().numpy()
                    z = ref_key_body_pos[i, j, 2].cpu().numpy()
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self._gym, self._viewer, self._envs[i], sphere_pose)
        else:
            pass
        
    def _draw_height_points_around_base(self):
        # draw height lines
        if not self._cfg.terrain.measure_heights:
            return
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 8, 8, None, color=(1, 1, 0))
        for i in range(self._num_envs):
            base_pos = (self._root_states[i, :3]).cpu().numpy()
            heights = self._measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self._base_quat[i].repeat(heights.shape[0]), self._height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self._gym, self._viewer, self._envs[i], sphere_pose)
    
    def _draw_height_points_around_feet(self):
        """ Draws height measurement points around feet for debugging
        """
        # Height points around feet
        height_points = torch.zeros(self._num_envs, 9*len(self._feet_indices), 3, device=self._device)
        feet_pos = self._rigid_body_states[:, self._feet_indices, 0:3]
        foot_points = feet_pos + self._cfg.terrain.border_size
        foot_points = (foot_points/self._cfg.terrain.horizontal_scale).long()
        px = foot_points[:, :, 0].view(-1)
        py = foot_points[:, :, 1].view(-1)
        heights1 = self._height_samples[px-1, py]  # [x-0.1, y]
        heights2 = self._height_samples[px+1, py]  # [x+0.1, y]
        heights3 = self._height_samples[px, py-1]  # [x, y-0.1]
        heights4 = self._height_samples[px, py+1]  # [x, y+0.1]
        heights5 = self._height_samples[px, py]    # [x, y]
        heights6 = self._height_samples[px-1, py-1]  # [x-0.1, y-0.1]
        heights7 = self._height_samples[px+1, py+1]  # [x+0.1, y+0.1]
        heights8 = self._height_samples[px-1, py+1]  # [x-0.1, y+0.1]
        heights9 = self._height_samples[px+1, py-1]  # [x+0.1, y-0.1]
        for i in range(len(self._feet_indices)):
            height_points[0, i*9+0, 0] = (px-1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+0, 1] = (py-1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+0, 2] = heights6.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
            height_points[0, i*9+1, 0] = (px-1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+1, 1] = py.view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+1, 2] = heights1.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
            height_points[0, i*9+2, 0] = px.view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+2, 1] = (py-1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+2, 2] = heights3.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
            height_points[0, i*9+3, 0] = px.view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+3, 1] = (py+1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+3, 2] = heights4.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
            height_points[0, i*9+4, 0] = px.view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+4, 1] = py.view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+4, 2] = heights5.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
            height_points[0, i*9+5, 0] = (px+1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+5, 1] = py.view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+5, 2] = heights2.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
            height_points[0, i*9+6, 0] = (px+1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+6, 1] = (py+1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+6, 2] = heights7.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
            height_points[0, i*9+7, 0] = (px-1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+7, 1] = (py+1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+7, 2] = heights8.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
            height_points[0, i*9+8, 0] = (px+1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+8, 1] = (py-1).view(self._num_envs, -1)[0, i] * self._cfg.terrain.horizontal_scale - self._cfg.terrain.border_size
            height_points[0, i*9+8, 2] = heights9.view(self._num_envs, -1)[0, i] * self._cfg.terrain.vertical_scale
        
        self._gym.clear_lines(self._viewer)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))
        for i in range(self._num_envs):
            for j in range(9*len(self._feet_indices)):
                x = height_points[i, j, 0].cpu().numpy()
                y = height_points[i, j, 1].cpu().numpy()
                z = height_points[i, j, 2].cpu().numpy()
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self._gym, self._viewer, self._envs[i], sphere_pose)
    
    def _draw_debug_boxes(self, pos, orientation):
        """
        Draws debug boxes at specified positions and orientations.
        Args:
            pos (torch.Tensor): Tensor of shape (num_envs, 3) specifying the positions of the boxes.
            orientation (torch.Tensor): Tensor of shape (num_envs, 4) specifying the orientations (quaternions) of the boxes.
        """
        self._gym.clear_lines(self._viewer)
        self._gym.refresh_rigid_body_state_tensor(self._sim)
        box_geom = gymutil.WireframeBoxGeometry(0.5, 0.25, 0.25, color=(0, 1, 0))
        for i in range(self._num_envs):
            box_pos = pos[i].cpu().numpy()
            box_quat = orientation[i].cpu().numpy()
            box_pose = gymapi.Transform(gymapi.Vec3(box_pos[0], box_pos[1], box_pos[2]),
                                        gymapi.Quat(box_quat[0], box_quat[1], box_quat[2], box_quat[3]))
            gymutil.draw_lines(box_geom, self._gym, self._viewer, self._envs[i], box_pose)
    
    def _draw_debug_depth_images(self):
        depth = self._depth_images[0, 0, :, :]  # get depth image of first env, first step
        # print(f"depth values: {depth}")
        if self._cfg.sensor.depth_camera_config.calculate_depth:
            # far_clip = self._cfg.sensor.depth_camera_config.far_clip
            # near_clip = self._cfg.sensor.depth_camera_config.near_clip
            pixel_values = ((depth + 0.5) * 255.0).cpu().numpy().astype(np.uint8)
            # image = im.fromarray(pixel_values, mode='L')
            # image.save("debug_depth_images/depth_frame%d.jpg" % self.frame_count)
            import cv2 as cv
            cv.imshow("Depth Camera", pixel_values)
            cv.waitKey(1)
            # pixel_values = (depth + 0.5) * (far_clip - near_clip) + near_clip
            # pixel_values = pixel_values.cpu().numpy().astype(np.float32)
            # print(f"depth pixel values: {pixel_values}")
        elif self._cfg.sensor.depth_camera_config.return_pointcloud:
            pointcloud = self._depth_images[0, 0, :, :, :]  # get pointcloud of first env, first step
            pointcloud_np = pointcloud.cpu().numpy().astype(np.float32)
            self._gym.clear_lines(self._viewer)
            self._gym.refresh_rigid_body_state_tensor(self._sim)
            for i in range(pointcloud_np.shape[0]):
                for j in range(pointcloud_np.shape[1]):
                    x = pointcloud_np[i, j, 0]
                    y = pointcloud_np[i, j, 1]
                    z = pointcloud_np[i, j, 2]
                    sphere_geom = gymutil.WireframeSphereGeometry(0.01, 4, 4, None, color=(0, 1, 1))
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self._gym, self._viewer, self._envs[0], sphere_pose)
            # print(f"pointcloud values: {pointcloud_np}")
    
    def _render(self, sync_frame_time=True):
        if self._viewer:
            # check for window closed
            if self._gym.query_viewer_has_closed(self._viewer):
                sys.exit()

            # check for keyboard events
            for evt in self._gym.query_viewer_action_events(self._viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self._enable_viewer_sync = not self._enable_viewer_sync

            # fetch results
            if self._device != 'cpu':
                self._gym.fetch_results(self._sim, True)

            # step graphics
            if self._enable_viewer_sync:
                self._gym.step_graphics(self._sim)
                self._gym.draw_viewer(self._viewer, self._sim, True)
                if sync_frame_time:
                    self._gym.sync_frame_time(self._sim)
            else:
                self._gym.poll_viewer_events(self._viewer)
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self._cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self._cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self._num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
            self._friction_values[env_id, :] = self.friction_coeffs[env_id]

        if self._cfg.domain_rand.randomize_restitution:
            if env_id == 0:
                restitution_range = self._cfg.domain_rand.restitution_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self._num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets, 1), device='cpu')
                self.restitution_coeffs = restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
            self._restitution_values[env_id, :] = self.restitution_coeffs[env_id]

        return props
    
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self._dof_pos_limits = torch.zeros(self._num_dof, 2, dtype=torch.float, device=self._device, requires_grad=False)
            self._dof_vel_limits = torch.zeros(self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)
            self._torque_limits = torch.zeros(self._num_dof, dtype=torch.float, device=self._device, requires_grad=False)
            for i in range(len(props)):
                self._dof_pos_limits[i, 0] = props["lower"][i].item()
                self._dof_pos_limits[i, 1] = props["upper"][i].item()
                self._dof_vel_limits[i] = props["velocity"][i].item()
                self._torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self._dof_pos_limits[i, 0] + self._dof_pos_limits[i, 1]) / 2
                r = self._dof_pos_limits[i, 1] - self._dof_pos_limits[i, 0]
                self._dof_pos_limits[i, 0] = m - 0.5 * r * self._cfg.rewards.soft_dof_pos_limit
                self._dof_pos_limits[i, 1] = m + 0.5 * r * self._cfg.rewards.soft_dof_pos_limit
        
        if self._cfg.domain_rand.randomize_joint_friction:
            joint_friction_range = np.array(
                self._cfg.domain_rand.joint_friction_range, dtype=np.float32)
            friction = np.random.uniform(
                joint_friction_range[0], joint_friction_range[1])
            self._joint_friction[env_id] = friction
            for j in range(self._num_dof):
                props["friction"][j] = torch.tensor(
                    friction, dtype=torch.float, device=self._device)

        if self._cfg.domain_rand.randomize_joint_damping:
            joint_damping_range = np.array(
                self._cfg.domain_rand.joint_damping_range, dtype=np.float32)
            damping = np.random.uniform(
                joint_damping_range[0], joint_damping_range[1])
            self._joint_damping[env_id] = damping
            for j in range(self._num_dof):
                props["damping"][j] = torch.tensor(
                    damping, dtype=torch.float, device=self._device)

        if self._cfg.domain_rand.randomize_joint_armature:
            joint_armature_range = np.array(
                self._cfg.domain_rand.joint_armature_range, dtype=np.float32)
            armature = np.random.uniform(
                joint_armature_range[0], joint_armature_range[1])
            self._joint_armature[env_id] = armature
            for j in range(self._num_dof):
                props["armature"][j] = torch.tensor(
                    armature, dtype=torch.float, device=self._device)
        else:
            if len(self._cfg.asset.dof_armature) != self._num_dof:
                warnings.warn("Use default armature 0.01 for all dofs. To specify dof armatures, please see dof_armature in legged_robot_config.py")
                for j in range(self._num_dof):
                    props["armature"][j] = torch.tensor(
                        0.01, dtype=torch.float, device=self._device)
            else: # use dof_armature in asset config
                for j in range(self._num_dof):
                    props["armature"][self._dof_indices[j]] = torch.tensor(
                        self._cfg.asset.dof_armature[j], dtype=torch.float, device=self._device)
        
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self._cfg.domain_rand.randomize_base_mass:
            rng = self._cfg.domain_rand.added_mass_range
            added_base_mass = np.random.uniform(rng[0], rng[1])
            props[0].mass += added_base_mass
            self._added_base_mass[env_id] = added_base_mass

        # randomize com position
        if self._cfg.domain_rand.randomize_com_displacement:
            com_x_bias = np.random.uniform(
                self._cfg.domain_rand.com_pos_x_range[0], self._cfg.domain_rand.com_pos_x_range[1])
            com_y_bias = np.random.uniform(
                self._cfg.domain_rand.com_pos_y_range[0], self._cfg.domain_rand.com_pos_y_range[1])
            com_z_bias = np.random.uniform(
                self._cfg.domain_rand.com_pos_z_range[0], self._cfg.domain_rand.com_pos_z_range[1])

            self._base_com_bias[env_id, 0] += com_x_bias
            self._base_com_bias[env_id, 1] += com_y_bias
            self._base_com_bias[env_id, 2] += com_z_bias

            # randomize com position of "base1_downbox"
            props[0].com.x += com_x_bias
            props[0].com.y += com_y_bias
            props[0].com.z += com_z_bias
            # print(f"com of base: {props[0].com} (after randomization)")
        
        return props
    
    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self._cfg.terrain.static_friction
        plane_params.dynamic_friction = self._cfg.terrain.dynamic_friction
        plane_params.restitution = self._cfg.terrain.restitution
        self._gym.add_ground(self._sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self._terrain.cfg.horizontal_scale
        hf_params.row_scale = self._terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self._terrain.cfg.vertical_scale
        hf_params.nbRows = self._terrain.tot_rows
        hf_params.nbColumns = self._terrain.tot_cols 
        hf_params.transform.p.x = -self._terrain.cfg.border_size 
        hf_params.transform.p.y = -self._terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self._cfg.terrain.static_friction
        hf_params.dynamic_friction = self._cfg.terrain.dynamic_friction
        hf_params.restitution = self._cfg.terrain.restitution
        self._gym.add_heightfield(self._sim, 
                                  self._terrain.heightsamples.transpose(), # column first order
                                  hf_params)
        self._height_samples = torch.tensor(self._terrain.heightsamples).view(self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)
        self._edge_mask = torch.tensor(self._terrain.edge_mask).view(self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)
    
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self._terrain.terrain_mesh.vertices.shape[0]
        tm_params.nb_triangles = self._terrain.terrain_mesh.faces.shape[0]

        # give a small offset (horizontal_scale/2) to the terrain position to align trimesh with heightfield
        tm_params.transform.p.x = -self._terrain.cfg.border_size - self._cfg.terrain.horizontal_scale / 2.0
        tm_params.transform.p.y = -self._terrain.cfg.border_size - self._cfg.terrain.horizontal_scale / 2.0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self._cfg.terrain.static_friction
        tm_params.dynamic_friction = self._cfg.terrain.dynamic_friction
        tm_params.restitution = self._cfg.terrain.restitution
        vertices = np.array(self._terrain.terrain_mesh.vertices, dtype=np.float32)
        triangles = np.array(self._terrain.terrain_mesh.faces, dtype=np.uint32)
        self._gym.add_triangle_mesh(self._sim, 
                                    vertices.flatten(order='K'), 
                                    triangles.flatten(order='K'), 
                                    tm_params)
        self._height_samples = torch.tensor(self._terrain.heightsamples).view(self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)
        self._edge_mask = torch.tensor(self._terrain.edge_mask).view(self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)
        
    #----- Properties -----#
    @property
    def feet_contact_indices(self):
        return self._feet_indices
    
    @property
    def feet_pos(self):
        return self._rigid_body_states[:, self._feet_indices, :3]

    @property
    def feet_vel(self):
        return self._rigid_body_states[:, self._feet_indices, 7:10]
    
    @property
    def feet_quat(self):
        return self._rigid_body_states[:, self._feet_indices, 3:7]
    
    @property
    def key_body_pos(self):
        return self._rigid_body_states[:, self._key_body_indices, :3]
    
    @property
    def dof_pos_limits(self):
        return self._dof_pos_limits[self._dof_indices]
    
    @property
    def dof_vel_limits(self):
        return self._dof_vel_limits[self._dof_indices]
    
    @property
    def dof_pos(self):
        return self._dof_pos[:, self._dof_indices]
    
    @property
    def dof_vel(self):
        return self._dof_vel[:, self._dof_indices]
    
    @property
    def last_dof_vel(self):
        return self._last_dof_vel[:, self._dof_indices]
    
    @property
    def torques(self):
        return self._torques[:, self._dof_indices]
    
    @property
    def torque_limits(self):
        return self._torque_limits[self._dof_indices]
    
    @property
    def default_dof_pos(self):
        return self._default_dof_pos[:, self._dof_indices]
    
    @property
    def dr_kp_scale(self):
        return dr_normalize(self._kp_scale[:, self._dof_indices],
                            self._cfg.domain_rand.kp_range[0],
                            self._cfg.domain_rand.kp_range[1])
    
    @property
    def dr_kd_scale(self):
        return dr_normalize(self._kd_scale[:, self._dof_indices],
                            self._cfg.domain_rand.kd_range[0],
                            self._cfg.domain_rand.kd_range[1])