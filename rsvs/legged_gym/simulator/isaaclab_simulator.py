from legged_gym import *
from legged_gym.simulator.simulator import Simulator
from PIL import Image as im
import torch
import numpy as np
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math_utils import *
import warp as wp
import trimesh
from legged_gym.warp.warp_cam import WarpCam

if SIMULATOR == "isaaclab":
    from isaaclab.app import AppLauncher
    import carb
    GROUND_PATH = "/World/ground"
    LIGHT_PATH = "/World/Light"

""" ********** IsaacLab Simulator ********** """
class IsaacLabSimulator(Simulator):
    """For IsaacLab Simulator, we access states of the robot through the Articulation object defined in _create_envs(), 
       but not create extra buffer to store these states.

    Args:
        Simulator (_type_): _description_
    """
    def __init__(self, cfg, sim_params: dict, device, headless):
        self._sim_params = sim_params
        super().__init__(cfg, sim_params, device, headless)
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
        self._last_base_lin_vel[:] = self._base_lin_vel[:]
        self._last_base_ang_vel[:] = self._base_ang_vel[:]
        self._last_feet_vel[:] = self._robot.data.body_link_vel_w[:, self.feet_indices, :3]
        actions = self._pre_simulator_step(actions)
        for _ in range(self._cfg.control.decimation):
            self._last_dof_vel[:] = self._robot.data.joint_vel[:] # for computing velocity-target-based PD control
            self._compute_torques(actions)
            self._robot.write_data_to_sim()
            self._sim.step(render=False)
            self._robot.update(self._sim_params["dt"])
            self._contact_sensors.update(self._sim_params["dt"])
        if not self._headless:
            self._sim.render()

    def post_physics_step(self):
        # Cache robot data reference to avoid repeated attribute lookups
        robot_data = self._robot.data
        self._base_pos[:] = robot_data.root_pos_w[:]
        self._check_base_pos_out_of_bound()       # check if the pos of the robot is out of terrain bounds
        self._base_pos[:] = robot_data.root_pos_w[:]
        # convert wxyz to xyzw
        root_quat = robot_data.root_quat_w
        self._base_quat[:] = root_quat[:, [1,2,3,0]] # convert from (w,x,y,z) to (x,y,z,w)
        self._base_euler[:] = get_euler_xyz(self._base_quat)
        self._projected_gravity[:] = quat_rotate_inverse(self._base_quat, self._global_gravity)[:]
        self._base_lin_vel[:] = quat_rotate_inverse(self._base_quat, robot_data.root_lin_vel_w)[:]
        self._base_ang_vel[:] = quat_rotate_inverse(self._base_quat, robot_data.root_ang_vel_w)[:]
        # Cache body link data to avoid multiple accesses
        body_link_pos = robot_data.body_link_pos_w
        body_link_vel = robot_data.body_link_vel_w
        self._key_body_pos[:] = body_link_pos[:, self._key_body_indices, :]
        self._feet_pos[:] = body_link_pos[:, self._feet_indices, :]
        self._feet_vel[:] = body_link_vel[:, self._feet_indices, :3]
        # Link contact state
        if self._cfg.asset.obtain_link_contact_states:
            # use the max value of history of contact forces to determine contact state
            self._link_contact_states = 1. * (torch.max(torch.norm(
                self._contact_sensors.data.net_forces_w_history[:, :, self._contact_state_link_indices, :], dim=-1), dim=1)[0] > 10.)
        # update terrain heights info
        if self._cfg.terrain.measure_heights:
            self._update_surrounding_heights()
            if self._cfg.terrain.obtain_terrain_info_around_feet:
                self._calc_terrain_info_around_feet()
    
    def reset_idx(self, env_ids):
        if self._cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(env_ids)
        if self._cfg.domain_rand.randomize_joint_friction:
            self._randomize_joint_friction(env_ids)
        if self._cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(env_ids)
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(env_ids)
        
        self._robot.reset(env_ids)
        self._contact_sensors.reset(env_ids)
        
        self._base_pos[:] = self._robot.data.root_pos_w[:]
        # convert wxyz to xyzw
        self._base_quat[:] = self._robot.data.root_quat_w[:, [1,2,3,0]] # convert from (w,x,y,z) to (x,y,z,w)
        self._base_euler[:] = get_euler_xyz(self._base_quat)
        self._projected_gravity[:] = quat_rotate_inverse(self._base_quat, self._global_gravity)[:]
        self._base_lin_vel[:] = quat_rotate_inverse(self._base_quat, self._robot.data.root_lin_vel_w)[:]
        self._base_ang_vel[:] = quat_rotate_inverse(self._base_quat, self._robot.data.root_ang_vel_w)[:]
        self._feet_pos[:] = self._robot.data.body_link_pos_w[:, self._feet_indices, :]
        self._key_body_pos[:] = self._robot.data.body_link_pos_w[:, self._key_body_indices, :]
        self._feet_vel[:] = self._robot.data.body_link_vel_w[:, self._feet_indices, :3]
        # Link contact state
        if self._cfg.asset.obtain_link_contact_states:
            # use the max value of history of contact forces to determine contact state
            self._link_contact_states = 1. * (torch.max(torch.norm(
                self._contact_sensors.data.net_forces_w_history[:, :, self._contact_state_link_indices, :], dim=-1), dim=1)[0] > 10.)
        # update terrain heights info
        if self._cfg.terrain.measure_heights:
            self._update_surrounding_heights()
            if self._cfg.terrain.obtain_terrain_info_around_feet:
                self._calc_terrain_info_around_feet()
        
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
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self._dof_indices, env_ids)
        
    def reset_root_states(self, 
                          env_ids, 
                          base_pos, 
                          base_quat, 
                          base_lin_vel_w, 
                          base_ang_vel_w):
        # base quat
        quat_sim = base_quat.clone()
        quat_sim[:] = base_quat[:, [3, 0, 1, 2]] # convert from (x,y,z,w) to (w,x,y,z)
        self._robot.write_root_state_to_sim(
            torch.cat((
                base_pos,
                quat_sim,
                base_lin_vel_w,
                base_ang_vel_w), dim=-1), 
            env_ids)
    
    def update_sensors(self):
        if self._cfg.sensor.add_depth:
            if self._depth_image_update_counter % self._depth_image_update_decimation == 0:
                self._update_depth_camera()
            self._depth_image_update_counter += 1
    
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
        max_push_vel_xy = self._cfg.domain_rand.max_push_vel_xy
        cur_vel_w = self._robot.data.root_vel_w
        push_vel = torch_rand_float(-max_push_vel_xy,
                                    max_push_vel_xy, (self._num_envs, 2), device=self._device)
        self._rand_push_vels[:, :2] = push_vel.detach().clone()
        cur_vel_w[:, :2] += push_vel
        self._robot.write_root_velocity_to_sim(cur_vel_w)
        self._last_base_lin_vel[:] = self._base_lin_vel[:]
        self._base_lin_vel[:] = quat_rotate_inverse(self._base_quat, self._robot.data.root_lin_vel_w)[:]
    
    def push_links(self):
        max_force = self._cfg.domain_rand.max_push_force
        # apply random forces to the links of the robot
        push_force = torch.rand((self._num_envs, self._num_bodies, 3), 
                                device=self._device) * 2 * max_force - max_force
        self._robot.instantaneous_wrench_composer.set_forces_and_torques(
            push_force, 
            torch.zeros_like(push_force), # zero torque
            is_global=True
        )
    
    def draw_debug_vis(self,
                       ref_key_body_pos=None):
        
        if self._cfg.env.debug_draw_key_body_points:
            self._draw_key_body_points(ref_key_body_pos)
        
    
    def set_viewer_camera(self, eye: np.ndarray, target: np.ndarray):
        self._sim.set_camera_view(eye=eye, 
                                  target=target)
    
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
        
        feet_pos_xy = self._feet_pos[:, :, :2] # (num_envs, num_feet, 2)
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
        self._control_dt = self._cfg.control.decimation * self._sim_params["dt"]
        if self._cfg.sensor.add_depth:
            self._num_camera_envs = self._cfg.env.num_camera_envs
            # update counter for depth images
            self._depth_image_update_counter = 0
            self._depth_image_update_decimation = self._cfg.sensor.depth_camera_config.decimation
            
    def _create_sim(self):
        self._app_launcher = AppLauncher({"headless": self._headless, "device": self._device})
        
        import isaaclab.sim as sim_utils
        from isaacsim.core.utils.stage import get_current_stage
        
        physx_cfg = sim_utils.PhysxCfg(
            solver_type=self._sim_params["physx"]["solver_type"],
            max_position_iteration_count=self._sim_params["physx"]["num_position_iterations"],
            max_velocity_iteration_count=self._sim_params["physx"]["num_velocity_iterations"],
            bounce_threshold_velocity=self._sim_params["physx"]["bounce_threshold_velocity"],
            gpu_max_rigid_contact_count=self._sim_params["physx"]["max_gpu_contact_pairs"],
        )
        # create simulation context
        sim_cfg = sim_utils.SimulationCfg(device=self._device, 
                                          dt=self._sim_params["dt"],
                                          render_interval=self._cfg.control.decimation,
                                          physx=physx_cfg)
        
        self._sim = sim_utils.SimulationContext(sim_cfg)
        self._stage = get_current_stage()
        
        # disable delays during rendering
        carb_settings = carb.settings.get_settings()
        carb_settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)

        # add terrain
        mesh_type = self._cfg.terrain.mesh_type
        if mesh_type == "plane":
            self._create_ground_plane()            
        elif mesh_type == "heightfield":
            raise NotImplementedError("Heightfield terrain not implemented for IsaacLabSimulator yet")
        elif mesh_type == "trimesh":
            self._terrain = Terrain(self._cfg.terrain)
            self._create_trimesh()
        else:
            raise NameError(f"Unknown terrain mesh type: {mesh_type}")
        
        # build lights
        if not self._headless:
            self._build_lights()
        
        # specify the boundary of the heightfield
        self._terrain_x_range = torch.zeros(2, device=self._device)
        self._terrain_y_range = torch.zeros(2, device=self._device)
        if self._cfg.terrain.mesh_type in ['heightfield', 'trimesh']:
            # give a small margin(1.0m)
            self._terrain_x_range[0] = -self._cfg.terrain.border_size + 1.0
            self._terrain_x_range[1] = self._cfg.terrain.border_size + \
                self._cfg.terrain.num_rows * self._cfg.terrain.terrain_length - 1.0
            self._terrain_y_range[0] = -self._cfg.terrain.border_size + 1.0
            self._terrain_y_range[1] = self._cfg.terrain.border_size + \
                self._cfg.terrain.num_cols * self._cfg.terrain.terrain_width - 1.0
        elif self._cfg.terrain.mesh_type == 'plane':  # the plane used has limited size,
            # and the origin of the world is at the center of the plane
            self._terrain_x_range[0] = -self._cfg.terrain.plane_length/2+1
            self._terrain_x_range[1] = self._cfg.terrain.plane_length/2-1
            # the plane is a square
            self._terrain_y_range[0] = -self._cfg.terrain.plane_length/2+1
            self._terrain_y_range[1] = self._cfg.terrain.plane_length/2-1
        
    def _create_envs(self):
        """ Creates environments, adds the robot asset to each environment, sets DOF properties and calls callbacks to process rigid shape, rigid body and DOF properties.
        """
        from isaacsim.core.cloner import Cloner
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.sensors import ContactSensorCfg, ContactSensor
        
        self._cloner = Cloner(self._stage)
        source_env_path = "/World/envs/env_0"
        prim_paths = self._cloner.generate_paths("/World/envs/env", self._num_envs)
        self._stage.DefinePrim(source_env_path, "Xform")
        
        if self._cfg.asset.name == "go2" or self._cfg.asset.name == "g1":
            solver_pos_iteration = 4
            solver_vel_iteration = 0
        elif self._cfg.asset.name == "k1":
            solver_pos_iteration = 8
            solver_vel_iteration = 4
        
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=not self._cfg.asset.self_collisions, 
                    fix_root_link=self._cfg.asset.fix_base_link,
                    solver_position_iteration_count=solver_pos_iteration,
                    solver_velocity_iteration_count=solver_vel_iteration,)
        
        rigid_props = sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=self._sim_params["physx"]["max_depenetration_velocity"],
                                                       angular_damping=self._cfg.asset.angular_damping,
                                                       linear_damping=self._cfg.asset.linear_damping,
                                                       disable_gravity=self._cfg.asset.disable_gravity,
                                                       max_linear_velocity=self._cfg.asset.max_linear_velocity,
                                                       max_angular_velocity=self._cfg.asset.max_angular_velocity)
        
        # Use urdf file to keep consistent with other simulators
        urdf_cfg = sim_utils.UrdfFileCfg(
            asset_path=self._cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),
            fix_base=self._cfg.asset.fix_base_link,
            merge_fixed_joints=self._cfg.asset.collapse_fixed_joints,
            replace_cylinders_with_capsules=self._cfg.asset.replace_cylinder_with_capsule,
            rigid_props=rigid_props,
            articulation_props=articulation_props,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
            activate_contact_sensors=True,
        )
        
        # convert xyzw to wxyz
        rot_sim = [self._cfg.init_state.rot[3], 
                   self._cfg.init_state.rot[0], 
                   self._cfg.init_state.rot[1], 
                   self._cfg.init_state.rot[2]]
        init_state = ArticulationCfg.InitialStateCfg(pos=self._cfg.init_state.pos, 
                                                     rot=rot_sim,
                                                     joint_pos=self._cfg.init_state.default_joint_angles)
        
        # specify actuator config based on the robot
        if self._cfg.asset.name == "go2":
            from resources.robots.unitree_robotics.go2.go2_lab_cfg import GO2_ACTUATOR_CFG
            actuator_cfg = GO2_ACTUATOR_CFG
        elif self._cfg.asset.name == "g1":
            from resources.robots.unitree_robotics.g1_description.g1_lab_cfg import G1_12DOF_ACTUATOR_CFG, G1_29DOF_ACTUATOR_CFG
            if len(self._cfg.asset.dof_names) == 12:
                actuator_cfg = G1_12DOF_ACTUATOR_CFG
            elif len(self._cfg.asset.dof_names) == 29:
                actuator_cfg = G1_29DOF_ACTUATOR_CFG
        elif self._cfg.asset.name == "k1":
            from resources.robots.booster_robotics.K1.k1_lab_cfg import K1_ACTUATOR_CFG
            actuator_cfg = K1_ACTUATOR_CFG
        else:
            raise NameError(f"Unknown robot name: {self._cfg.asset.name}")
        
        # create the first prim of env 0, then clone other envs
        articulation_cfg = ArticulationCfg(
                prim_path=f"/World/envs/env_.*/{self._cfg.asset.name}",
                spawn=urdf_cfg,
                init_state=init_state,
                actuator_value_resolution_debug_print=False,
                actuators=actuator_cfg,
                collision_group=0,
                soft_joint_pos_limit_factor=self._cfg.rewards.soft_dof_pos_limit
            )
        self._robot = Articulation(articulation_cfg)
        # clone other envs
        self._cloner.clone(source_prim_path=f"/World/envs/env_0",
                           prim_paths=prim_paths,
                           copy_from_source=False,
                           replicate_physics=True,
                           base_env_path=f"/World/envs",
                           enable_env_ids=True)
        
        # Add contact sensors
        contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/" + self._cfg.asset.name + "/.*", # track all links of the robot, but only the ones specified in cfg will be used for termination and penalty
            update_period=self._sim_params["dt"],   # update every physics step
            history_length=3,                       # keep contact history of last 3 steps
            debug_vis=not self._headless,           # visualize contact points if not headless
        )
        
        self._contact_sensors = ContactSensor(contact_sensor_cfg)
        
        # filter collisions between envs
        from pxr import PhysxSchema

        physics_scene_path = None
        for prim in self._stage.Traverse():
            if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                physics_scene_path = prim.GetPrimPath().pathString
                break

        if (physics_scene_path is None):
            assert(False), "No physics scene found! Please make sure one exists."
        
        env_prim_paths = [f"/World/envs/env_{i}" for i in range(self._num_envs)]
        self._cloner.filter_collisions(physics_scene_path, "/World/collisions",
                                       env_prim_paths, global_paths=[GROUND_PATH])
        
        # reset the simulation to make sure everything is initialized
        self._sim.reset()
        
        # print info after reset the simulation, to make sure the sensors are initialized
        # links in contact sensors have different order from the body names in the robot articulation, 
        # so we need to print them to make sure we get the correct indices for termination and penalty
        print(f"Created contact sensors: {self._contact_sensors}")
        
        self._get_env_origins()
        
        self._dof_names = self._robot.joint_names
        print(f"All dof names: {self._dof_names}")
        # find the indices (in the robot's joint list) of joints specified in self._cfg.asset.dof_names
        self._dof_indices = [self._dof_names.index(name) for name in self._cfg.asset.dof_names]
        print(f"dof indices: {self._dof_indices}")
        self._num_dof = len(self._dof_names)
        self._num_bodies = len(self._robot.body_names)
        
        def find_link_contact_indices(names: list[str]) -> list[int]:
            """find link indices in bodies of contact sensors based on link names specified in the config for termination and penalty.

            Args:
                names (list[str]): List of link names to find indices for

            Returns:
                list[int]: List of indices corresponding to the given link names
            """
            link_indices = list()
            for link in self._contact_sensors.body_names:
                flag = False
                for name in names:
                    if name in link:
                        flag = True
                if flag:
                    link_indices.append(self._contact_sensors.body_names.index(link))
            return link_indices
        
        def find_link_indices(names: list[str]) -> list[int]:
            """find link indices in bodies of the robot based on link names specified in the config for feet

            Args:
                names (list[str]): List of link names to find indices for
            Returns:
                list[int]: List of indices corresponding to the given link names
            """
            link_indices = list()
            for link in self._robot.body_names:
                flag = False
                for name in names:
                    if name in link:
                        flag = True
                if flag:
                    link_indices.append(self._robot.body_names.index(link))
            return link_indices

        self._termination_contact_indices = find_link_contact_indices(
            self._cfg.asset.terminate_after_contacts_on)
        print(f"All link names: {self._robot.body_names}")
        print(f"Termination contact link indices: {self._termination_contact_indices}")
        self._penalized_contact_indices = find_link_contact_indices(
            self._cfg.asset.penalize_contacts_on)
        print(f"Penalized contact link indices: {self._penalized_contact_indices}")
        self._feet_names = [
            link for link in self._robot.body_names if self._cfg.asset.foot_name in link
        ]
        # the order of bodies in contact sensors is different from the order of bodies in the robot articulation, so we need to find indices separately
        self._feet_contact_indices = find_link_contact_indices(self._feet_names)
        self._feet_indices = find_link_indices(self._feet_names)
        print(f"feet names: {self._feet_names}")
        assert len(self._feet_indices) > 0
        # get base link index in the robot articulation
        self._base_link_index = self._robot.body_names.index(self._cfg.asset.base_link_name)
        self._key_body_indices = find_link_indices(self._cfg.asset.key_bodies)
        print(f"key_body_indices: {self._key_body_indices}")
        
        if self._cfg.asset.obtain_link_contact_states:
            self._contact_state_link_indices = find_link_indices(
                self._cfg.asset.contact_state_link_names
            )
        
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        self._key_point_visualizer = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/Markers",
                markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )
                }
            )
            )
        self._init_domain_params()
        # randomize friction (only at startup; doing it on reset slows down training)
        if self._cfg.domain_rand.randomize_friction:
            self._randomize_friction(torch.arange(self._num_envs))
        # randomize restitution (only at startup; doing it on reset slows down training)
        if self._cfg.domain_rand.randomize_restitution:
            self._randomize_restitution(torch.arange(self._num_envs))
        # randomize base mass (only at startup; doing it on reset slows down training)
        if self._cfg.domain_rand.randomize_base_mass:
            self._randomize_base_mass(torch.arange(self._num_envs))
        # randomize COM displacement (only at startup; doing it on reset slows down training)
        if self._cfg.domain_rand.randomize_com_displacement:
            self._randomize_com_displacement(torch.arange(self._num_envs))
        # randomize joint armature
        if self._cfg.domain_rand.randomize_joint_armature:
            self._randomize_joint_armature(torch.arange(self._num_envs))
        # randomize joint friction
        if self._cfg.domain_rand.randomize_joint_friction:
            self._randomize_joint_friction(torch.arange(self._num_envs))
        # randomize joint damping
        if self._cfg.domain_rand.randomize_joint_damping:
            self._randomize_joint_damping(torch.arange(self._num_envs))
        # randomize pd gain
        if self._cfg.domain_rand.randomize_pd_gain:
            self._randomize_pd_gain(torch.arange(self._num_envs))
        
    def _init_buffers(self):
        self._base_pos = torch.zeros_like(self._robot.data.root_pos_w)
        self._base_lin_vel = torch.zeros_like(self._robot.data.root_lin_vel_b)
        self._base_ang_vel = torch.zeros_like(self._robot.data.root_ang_vel_b)
        self._last_dof_vel = torch.zeros_like(self._robot.data.joint_vel)
        self._last_base_lin_vel = torch.zeros_like(self._robot.data.root_lin_vel_b)
        self._last_base_ang_vel = torch.zeros_like(self._robot.data.root_ang_vel_b)
        self._p_gains = torch.zeros(self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self._d_gains = torch.zeros(self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
        self._base_quat = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float)
        self._base_euler = get_euler_xyz(self._base_quat)
        self._projected_gravity = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float)
        self._global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self._device, dtype=torch.float).repeat(self._num_envs, 1)
        self._base_init_quat = torch.tensor(
            self._cfg.init_state.rot, device=self._device
        )
        self._feet_pos = torch.zeros_like(self._robot.data.body_link_pos_w[:, self._feet_indices, :])
        self._feet_vel = torch.zeros_like(self._robot.data.body_link_vel_w[:, self._feet_indices, :3])
        self._key_body_pos = torch.zeros_like(self._robot.data.body_link_pos_w[:, self._key_body_indices, :])
        self._last_feet_vel = torch.zeros_like(self._robot.data.body_link_vel_w[:, self._feet_indices, :3])
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
        
        # Terrain information around feet
        if self._cfg.terrain.obtain_terrain_info_around_feet:
            self._normal_vector_around_feet = torch.zeros(
                self._num_envs, len(self.feet_indices) * 3, dtype=torch.float, device=self._device, requires_grad=False)
            self._height_around_feet = torch.zeros(
                self._num_envs, len(self.feet_indices), 9, dtype=torch.float, device=self._device, requires_grad=False)
        
        if self._cfg.asset.obtain_link_contact_states:
            self._link_contact_states = torch.zeros(
                self._num_envs, len(self._contact_state_link_indices), dtype=torch.float, device=self._device, requires_grad=False)
        
        # joint positions offsets and PD gains
        for i in range(self._num_dof):
            name = self._dof_names[i]
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
        print(f"p_gains: {self._p_gains}")
        
        # control delay
        if self._cfg.domain_rand.randomize_ctrl_delay:
            self._action_queue = torch.zeros(
                self._num_envs, self._cfg.domain_rand.ctrl_delay_step_range[1]+1, self._num_actions, dtype=torch.float, device=self._device, requires_grad=False)
            self._action_delay = torch.randint(self._cfg.domain_rand.ctrl_delay_step_range[0],
                                              self._cfg.domain_rand.ctrl_delay_step_range[1]+1, (self._num_envs,), device=self._device, requires_grad=False)
        
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
            self._env_origins = torch.zeros(
                self._num_envs, 3, device=self._device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self._cfg.terrain.max_init_terrain_level
            if not self._cfg.terrain.curriculum:
                max_init_level = self._cfg.terrain.num_rows - 1
            self._terrain_levels = torch.randint(
                0, max_init_level+1, (self._num_envs,), device=self._device)
            self._terrain_types = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
            if hasattr(self._cfg.env, "num_camera_envs"): # split envs with cameras to all terrain types, ensuring the diversity of data collected by cameras
                self._terrain_types[:self._cfg.env.num_camera_envs] = torch.div(
                    torch.arange(self._cfg.env.num_camera_envs, device=self._device), (self._cfg.env.num_camera_envs/self._cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
                self._terrain_types[self._cfg.env.num_camera_envs:] = torch.div(
                    torch.arange(self._num_envs - self._cfg.env.num_camera_envs, device=self._device), ((self._num_envs - self._cfg.env.num_camera_envs)/self._cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            else:
                self._terrain_types = torch.div(torch.arange(self._num_envs, device=self._device), (self._num_envs/self._cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self._max_terrain_level = self._cfg.terrain.num_rows
            self._terrain_origins = torch.from_numpy(
                self._terrain.env_origins).to(self._device).to(torch.float)
            self._env_origins[:] = self._terrain_origins[self._terrain_levels,
                                                       self._terrain_types]
        else:
            self._custom_origins = False
            self._env_origins = torch.zeros(
                self._num_envs, 3, device=self._device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self._num_envs))
            num_rows = np.ceil(self._num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(
                num_rows), torch.arange(num_cols), indexing='ij')
            # plane has limited size, we need to specify spacing base on num_envs, to make sure all robots are within the plane
            # restrict envs to a square of [plane_length/2, plane_length/2]
            spacing = self._cfg.env.env_spacing
            if num_rows * self._cfg.env.env_spacing > self._cfg.terrain.plane_length / 2 or \
                    num_cols * self._cfg.env.env_spacing > self._cfg.terrain.plane_length / 2:
                spacing = min((self._cfg.terrain.plane_length / 2) / (num_rows-1),
                              (self._cfg.terrain.plane_length / 2) / (num_cols-1))
            self._env_origins[:, 0] = spacing * xx.flatten()[:self._num_envs]
            self._env_origins[:, 1] = spacing * yy.flatten()[:self._num_envs]
            self._env_origins[:, 2] = 0.
            self._env_origins[:, 0] -= self._cfg.terrain.plane_length / 4
            self._env_origins[:, 1] -= self._cfg.terrain.plane_length / 4
        
    def _update_surrounding_heights(self):
        if self._cfg.terrain.mesh_type == 'plane':
            self._measured_heights = torch.zeros(self._num_envs, self._num_height_points, device=self._device, requires_grad=False)
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
        """ Finds neighboring points around each foot for terrain height measurement."""
        # Foot positions
        foot_points = self._feet_pos + self._cfg.terrain.border_size
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
        base_pos = self._base_pos.clone().to(self._device)
        x_out_of_bound = (base_pos[:, 0] >= self._terrain_x_range[1]) | (
            base_pos[:, 0] <= self._terrain_x_range[0])
        y_out_of_bound = (base_pos[:, 1] >= self._terrain_y_range[1]) | (
            base_pos[:, 1] <= self._terrain_y_range[0])
        out_of_bound_buf = x_out_of_bound | y_out_of_bound
        env_ids = out_of_bound_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        else:
            # reset base position to initial position
            base_pos[env_ids] = self._robot.data.default_root_state[env_ids, :3]
            base_pos[env_ids] += self._env_origins[env_ids]
            self._robot.write_root_pose_to_sim(
                torch.cat([base_pos[env_ids], 
                            self._robot.data.root_quat_w[env_ids]], dim=-1), 
                env_ids=env_ids)
    
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
        # convert from specified joint order to simulator's joint order
        if control_type=="P":
            torques = self._kp_scale * self._p_gains * \
                (actions_scaled[:, self._dof_indices] + self._robot.data.default_joint_pos \
                    - self._robot.data.joint_pos) \
                    - self._kd_scale * self._d_gains*self._robot.data.joint_vel
        elif control_type=="V":
            torques = self._kp_scale * self._p_gains * \
                (actions_scaled[:, self._dof_indices] - self._robot.data.joint_vel) - \
                    self._kd_scale * self._d_gains * self._robot.data.joint_acc
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        # print(f"torques applied: {torques[0]}")
        # sequence of torque is the same of as the DOF order in the robot articulation
        self._robot.set_joint_effort_target(
                torch.clip(torques, -self.torque_limits, self.torque_limits)
            )
        return
    
    def _init_domain_params(self):
        self._friction_values = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._restitution_values = torch.zeros(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._added_base_mass = torch.ones(
            self._num_envs, 1, dtype=torch.float, device=self._device, requires_grad=False)
        self._rand_push_vels = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        # save original COM position for the first time
        body_coms = self._robot.root_physx_view.get_coms().to(self._device)
        self._default_com_pos = torch.zeros(
            self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self._default_com_pos[:] = body_coms[:, self._base_link_index, :3]
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
        if len(env_ids) == 0:
            return
        min_friction, max_friction = self._cfg.domain_rand.friction_range

        # refer to https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.set_material_properties
        # All shapes in the same env have the same static friction and dynamic friction values
        friction_ratios = torch.rand((len(env_ids), 1, 1), 
                            dtype=torch.float,
                            device=self._device).repeat(1, self._robot.root_physx_view.max_shapes, 2) \
                            * (max_friction - min_friction) + min_friction
        # save values to domain randomization params
        self._friction_values[env_ids] = friction_ratios[:,0,0].unsqueeze(1).detach().clone()
        
        raw_material_props = self._robot.root_physx_view.get_material_properties().to(self._device)
        target_material_props = raw_material_props.clone()
        target_material_props[env_ids, :, 0:2] = friction_ratios[:]
        all_indices = torch.arange(self._robot.root_physx_view.count, device="cpu")
        # tensors passed to set_material_properties must be on CPU
        self._robot.root_physx_view.set_material_properties(
            target_material_props.to('cpu'), all_indices
        )

    def _randomize_restitution(self, env_ids):
        if len(env_ids) == 0:
            return
        min_restitution, max_restitution = self._cfg.domain_rand.restitution_range

        restitution_ratios = torch.rand((len(env_ids), 1, 1),
                            dtype=torch.float,
                            device=self._device).repeat(1, self._robot.root_physx_view.max_shapes, 1) \
                            * (max_restitution - min_restitution) + min_restitution
        self._restitution_values[env_ids] = restitution_ratios[:, 0, 0].unsqueeze(1).detach().clone()

        raw_material_props = self._robot.root_physx_view.get_material_properties().to(self._device)
        target_material_props = raw_material_props.clone()
        target_material_props[env_ids, :, 2:3] = restitution_ratios[:]
        all_indices = torch.arange(self._robot.root_physx_view.count, device="cpu")
        self._robot.root_physx_view.set_material_properties(
            target_material_props.to('cpu'), all_indices
        )

    def _randomize_base_mass(self, env_ids):
        if len(env_ids) == 0:
            return
        min_mass, max_mass = self._cfg.domain_rand.added_mass_range
        # refer to https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.set_masses
        added_mass = torch.rand((len(env_ids), 1), 
                                dtype=torch.float) * \
                        (max_mass - min_mass) + min_mass
        self._added_base_mass[env_ids] = added_mass[:].to(self._device).detach().clone()
        raw_mass = self._robot.root_physx_view.get_masses()
        base_mass_after_dr = self._robot.data.default_mass[env_ids.to('cpu'), self._base_link_index].unsqueeze(1)\
                            + added_mass
        mass_after_dr = raw_mass.clone()
        mass_after_dr[env_ids, self._base_link_index] = base_mass_after_dr.squeeze(1)
        all_indices = torch.arange(self._robot.root_physx_view.count, device="cpu")
        # tensors passed to set_masses must be on CPU
        self._robot.root_physx_view.set_masses(mass_after_dr.to("cpu"), all_indices)
        
    def _randomize_com_displacement(self, env_ids):
        if len(env_ids) == 0:
            return
        min_displacement_x, max_displacement_x = self._cfg.domain_rand.com_pos_x_range
        min_displacement_y, max_displacement_y = self._cfg.domain_rand.com_pos_y_range
        min_displacement_z, max_displacement_z = self._cfg.domain_rand.com_pos_z_range
        
        com_displacement = torch.zeros((len(env_ids), 1, 3), dtype=torch.float, device=self._device)
        com_displacement[:, 0, 0] = torch.rand((len(env_ids), 1), dtype=torch.float, device=self._device).squeeze(1) \
            * (max_displacement_x - min_displacement_x) + min_displacement_x
        com_displacement[:, 0, 1] = torch.rand((len(env_ids), 1), dtype=torch.float, device=self._device).squeeze(1) \
            * (max_displacement_y - min_displacement_y) + min_displacement_y
        com_displacement[:, 0, 2] = torch.rand((len(env_ids), 1), dtype=torch.float, device=self._device).squeeze(1) \
            * (max_displacement_z - min_displacement_z) + min_displacement_z
        self._base_com_bias[env_ids] = com_displacement[:, 0, :].detach().clone()

        base_com_after_dr = self._default_com_pos[env_ids] + com_displacement[:, 0, :]
        # refer to https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.set_coms
        raw_coms = self._robot.root_physx_view.get_coms().to(self._device)
        com_after_dr = raw_coms.clone()
        com_after_dr[env_ids, self._base_link_index, :3] = base_com_after_dr[:]
        all_indices = torch.arange(self._robot.root_physx_view.count, device="cpu")
        # tensors passed to set_coms must be on CPU
        self._robot.root_physx_view.set_coms(com_after_dr.to("cpu"), all_indices)
    
    def _randomize_joint_armature(self, env_ids):
        if len(env_ids) == 0:
            return
        
        min_armature, max_armature = self._cfg.domain_rand.joint_armature_range
        armature = torch.rand((len(env_ids),), dtype=torch.float, device=self._device) \
            * (max_armature - min_armature) + min_armature
        # save values to domain randomization params
        self._joint_armature[env_ids, 0] = armature.detach().clone()
        # [len(env_ids)] -> [len(env_ids), num_actions], all joints within the same env have the same armature
        armature = armature.unsqueeze(1).repeat(1, self._num_actions)
        # refer to https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.Articulation.write_joint_armature_to_sim
        self._robot.write_joint_armature_to_sim(armature, self._dof_indices, env_ids)
    
    def _randomize_joint_friction(self, env_ids):
        if len(env_ids) == 0:
            return
        
        min_friction, max_friction = self._cfg.domain_rand.joint_friction_range
        friction = torch.rand((len(env_ids), 1), dtype=torch.float, device=self._device) \
            * (max_friction - min_friction) + min_friction
        self._joint_friction[env_ids] = friction.detach().clone()
        # All joints within the same env have the same friction
        friction = friction.repeat(1, self._num_actions)
        # refer to https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.Articulation.write_joint_friction_coefficient_to_sim
        self._robot.write_joint_friction_coefficient_to_sim(
            friction, None, None, self._dof_indices, env_ids) # currently, only static friction coefficients are considered
        
    def _randomize_joint_damping(self, env_ids):
        if len(env_ids) == 0:
            return
        
        min_damping, max_damping = self._cfg.domain_rand.joint_damping_range
        damping = torch.rand((len(env_ids), 1), dtype=torch.float, device=self._device) \
            * (max_damping - min_damping) + min_damping
        self._joint_damping[env_ids] = damping.detach().clone()
        damping = damping.repeat(1, self._num_actions)
        # refer to https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.Articulation.write_joint_damping_to_sim
        self._robot.write_joint_damping_to_sim(damping, self._dof_indices, env_ids)
        
    def _randomize_pd_gain(self, env_ids):
        self._kp_scale[env_ids] = torch_rand_float(
                self._cfg.domain_rand.kp_range[0], self._cfg.domain_rand.kp_range[1], (len(env_ids), self._num_actions), device=self._device)
        self._kd_scale[env_ids] = torch_rand_float(
                self._cfg.domain_rand.kd_range[0], self._cfg.domain_rand.kd_range[1], (len(env_ids), self._num_actions), device=self._device)
    
    def _update_depth_camera(self):
        """ Renders the depth camera and retrieves the depth images
        """
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
      # extract terrain mesh
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
        self._sensor_pos_tensor = torch.zeros_like(self._base_pos[:self._num_camera_envs, :3])
        self._sensor_quat_tensor = torch.zeros_like(self._base_quat[:self._num_camera_envs, :4])

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
                -self._cfg.domain_rand.camera_euler_offset_range[0],
                self._cfg.domain_rand.camera_euler_offset_range[0],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          self._camera_euler_offset[:self._num_camera_envs, 1] = torch_rand_float(
                -self._cfg.domain_rand.camera_euler_offset_range[1],
                self._cfg.domain_rand.camera_euler_offset_range[1],
                (self._num_camera_envs,1), device=self._device).squeeze(1)
          self._camera_euler_offset[:self._num_camera_envs, 2] = torch_rand_float(
                -self._cfg.domain_rand.camera_euler_offset_range[2],
                self._cfg.domain_rand.camera_euler_offset_range[2],
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
    
    def _create_ground_plane(self):
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
            
        ground_col = np.array([1.0, 0.9, 0.75])
        ground_col *= 0.017
        ground_path = GROUND_PATH

        physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=self._cfg.terrain.static_friction, 
                                                          dynamic_friction=self._cfg.terrain.dynamic_friction,
                                                          restitution=self._cfg.terrain.restitution)
        plane_cfg = GroundPlaneCfg(physics_material=physics_material, 
                                   color=ground_col, 
                                   size=(self._cfg.terrain.plane_length, self._cfg.terrain.plane_length))
        self._terrain = spawn_ground_plane(prim_path=ground_path, cfg=plane_cfg)

    def _create_trimesh(self):
        from isaaclab.terrains.utils import create_prim_from_mesh
        import isaaclab.sim as sim_utils
        
        
        visual_material = sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        )
        
        physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=self._cfg.terrain.static_friction, 
                                                          dynamic_friction=self._cfg.terrain.dynamic_friction,
                                                          restitution=self._cfg.terrain.restitution)
        create_prim_from_mesh(GROUND_PATH + "/terrain", 
                              self._terrain.terrain_mesh, 
                              physics_material=physics_material,
                              visual_material=visual_material,
                              translation=(
                                  -self._cfg.terrain.border_size - self._cfg.terrain.horizontal_scale / 2.0,
                                  -self._cfg.terrain.border_size - self._cfg.terrain.horizontal_scale / 2.0,
                                  0.0
                              )
                              )
        
        self._height_samples = torch.tensor(self._terrain.heightsamples).view(
            self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)
        self._edge_mask = torch.tensor(self._terrain.edge_mask).view(self._terrain.tot_rows, self._terrain.tot_cols).to(self._device)
    
    def _build_lights(self):
        import isaaclab.sim as sim_utils
        import isaacsim.core.utils.prims as prim_utils
        from pxr import Gf

        light_quat = quat_from_euler_xyz(torch.tensor(0.7),
                                        torch.tensor(0.0), 
                                        torch.tensor(0.6))
        light_quat = light_quat.tolist()
        distant_light_path = LIGHT_PATH + "/distant_light_xform"
        light_xform = prim_utils.create_prim(distant_light_path, "Xform")

        gf_quatf = Gf.Quatd()
        gf_quatf.SetReal(light_quat[-1])
        gf_quatf.SetImaginary(tuple(light_quat[:-1]))
        light_xform.GetAttribute("xformOp:orient").Set(gf_quatf)

        distant_light_cfg = sim_utils.DistantLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        self._distant_light = distant_light_cfg.func(distant_light_path + "/distant_light", distant_light_cfg)

        dome_light_cfg = sim_utils.DomeLightCfg(intensity=800.0, color=(0.7, 0.7, 0.7))
        self._dome_light = dome_light_cfg.func(LIGHT_PATH + "/dome_light", dome_light_cfg)
    
    def _draw_key_body_points(self, ref_key_body_pos=None):
        if ref_key_body_pos is None:
            pass
        else:
            self._key_point_visualizer.visualize(ref_key_body_pos.view(-1, 3))
    
    #----- Properties -----#
    @property
    def dof_pos_limits(self):
        """Returns the DOF position limits of the robot.

        Returns:
            Tensor ((num_dof, 2)): DOF position limits of the robot.
        """
        return self._robot.data.soft_joint_pos_limits[0, self._dof_indices, :]
    
    @property
    def dof_vel_limits(self):
        """Returns the DOF velocity limits of the robot.

        Returns:
            Tensor ((num_dof,)): DOF velocity limits of the robot.
        """
        return self._robot.data.joint_vel_limits[0, self._dof_indices]
    
    @property
    def base_init_pos(self):
        """Returns the initial base position of the robot.

        Returns:
            Tensor ((3,)): Initial base position of the robot.
        """
        return self._robot.data.default_root_state[0, :3]
    
    @property
    def dof_pos(self):
        """Returns the DOF positions of the robot.

        Returns:
            Tensor: DOF positions of the robot.
        """
        return self._robot.data.joint_pos[:, self._dof_indices]
    
    @property
    def dof_vel(self):
        """Returns the DOF velocities of the robot.

        Returns:
            Tensor: DOF velocities of the robot.
        """
        return self._robot.data.joint_vel[:, self._dof_indices]
    
    @property
    def last_dof_vel(self):
        """Returns the DOF velocities of the robot in the last simulation step.

        Returns:
            Tensor: DOF velocities of the robot in the last simulation step.
        """
        return self._last_dof_vel[:, self._dof_indices]
    
    @property
    def link_contact_forces(self):
        """Returns the contact forces of all links of the robot.

        Returns:
            Tensor: Contact forces of all links of the robot.
        """
        return self._contact_sensors.data.net_forces_w_history
    
    @property
    def feet_quat(self):
        feet_quat_sim = self._robot.data.body_link_quat_w[:, self._feet_indices, :]
        return feet_quat_sim[..., [1, 2, 3, 0]] # convert from (w,x,y,z) to (x,y,z,w)
    
    @property
    def torques(self):
        """Returns the torques applied to the robot's joints.

        Returns:
            Tensor: Torques applied to the robot's joints.
        """
        # convert from simulation order to specified order
        return self._robot.data.computed_torque[:, self._dof_indices]
    
    @property
    def torque_limits(self):
        """Returns the torque limits of the robot's joints.

        Returns:
            Tensor: Torque limits of the robot's joints.
        """
        return self._robot.data.joint_effort_limits[0, self._dof_indices]
    
    @property
    def default_dof_pos(self):
        """Returns the default dof pos.
        
        Returns:
            Tensor: (1, num_dofs)
        """
        return self._robot.data.default_joint_pos[0, self._dof_indices].unsqueeze(0)
    
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