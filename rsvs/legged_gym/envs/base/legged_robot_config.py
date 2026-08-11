from typing import Dict, List, Tuple
from .base_config import BaseConfig

# Type aliases for common config patterns
RewardScales = Dict[str, float]
PDGains = Dict[str, float]

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs: int = 4096 # number of parallel environments
        num_observations: int = 48 # size of the observation vector
        num_privileged_obs: int = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for asymmetric training). 
                                           # None is returned otherwise 
        num_actions: int = 12 # size of the action vector
        send_timeouts: bool = True # send time out information to the algorithm
        episode_length_s: int = 20 # episode length in seconds
        env_spacing: float = 2.0 # spacing between envs in the scene, only for plane
        fail_to_terminal_time_s: float = 0.1 # time before a fail state leads to environment reset, refer to https://github.com/limxdynamics/tron1-rl-isaacgym/tree/master
        debug: bool = False # enable debug drawings in the simulator
        debug_draw_height_points_around_base: bool = False # obtain height measurements around the base
        debug_draw_height_points_around_feet: bool = False # obtain height measurements around the feet (9 points around each foot, see terrain.measured_points_x/y)
        debug_draw_terrain_height_points: bool = False # draw all height points of the terrain
        debug_draw_terrain_edge_points: bool = False # draw edge points of the terrain, which can be used for reward design
        debug_draw_key_body_points: bool = False # draw key body points for mimic tasks
        debug_draw_depth_images: bool = False # draw depth images from the depth camera sensors
        max_projected_gravity: float = -0.1 # max allowed projected gravity in z axis
        
    class terrain:
        
        # heightfield uses a grid of height samples to represent the terrain, creating enormous points
        # trimesh creates terrain mesh directly, reducing the number of triangles compared with heightfield
        mesh_type: str = 'plane' # plane, heightfield, trimesh
        simplify_mesh: bool = True # whether to simplify the mesh by merging coplanar triangles, only for trimesh
        plane_length: float = 200.0 # [m]. plane size is 200x200x10 by default
        horizontal_scale: float = 0.1 # [m] distance between height samples in x and y direction
        vertical_scale: float = 0.005 # [m] distance between height samples in z direction
        border_size: int = 5 # [m] length of the border surrounding the terrain
        border_height: float = 1.0 # [m] height of the border surrounding the terrain
        curriculum: bool = False # whether to use terrain curriculum, starting from easier terrains and gradually increasing the difficulty
        static_friction: float = 1.0 # coefficient of static friction of the terrain
        dynamic_friction: float = 1.0 # coefficient of dynamic friction of the terrain
        restitution: float = 0.0 # coefficient of restitution of the terrain
        # rough terrain only:
        # obtain terrain height information around feet (default: 9 points around feet), measure_
        # x  x   x
        # x F(x) x
        # x  x   x (x: height point, F: foot position)
        obtain_terrain_info_around_feet: bool = False
        measure_heights: bool = False # obtain height measurements
        # positions of the sampling height around the base (relative to the base of the robot)
        measured_points_x: List[float] = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y: List[float] = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected: bool = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level: int = 1 # starting curriculum level
        terrain_length: float = 6.0 # [m] length of each subterrain, X direction
        terrain_width: float = 6.0 # [m] width of each subterrain, Y direction
        platform_size: float = 3.0 # [m] size of the flat platform at the center of each subterrain
        num_rows: int = 4  # number of terrain rows (levels), X direction
        num_cols: int = 4  # number of terrain cols (types), Y direction
        # terrain types: [smooth slope, random uniform, stairs up, stairs down, discrete]
        terrain_proportions: List[float] = [0.1, 0.1, 0.35, 0.25, 0.2]
        # difficulty scaling of the terrain parameters, the actual parameters will be computed by eval() with difficulty as the variable in make_terrain() function
        terrain_curriculum_difficulty: Dict[str, str] = {
            "slope": "difficulty * 0.4", # max slope in radians, will be multiplied by curriculum difficulty
            "step_height": "0.05 + 0.2 * difficulty", 
            "discrete_height": "0.05 + 0.2 * difficulty",
            "stepping_stones_params": {
                "stone_length": "np.random.uniform(1.6, 2.0)",
                "stone_width": "np.random.uniform(1.0, 2.0)",
                "stone_distance_x": "0.1 + 0.8 * difficulty",
                "stone_distance_y": "np.random.uniform(0.3, 0.5)",
                "max_height": "0"
            },
            "gap_size": "0.1 + difficulty * 0.8",
            "pit_depth": "0.1 + 0.6 * difficulty",
        }
        # trimesh only:
        slope_treshold: float = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state:
        pos: List[float] = [0.0, 0.0, 1.0] # x,y,z [m]
        # [Convention] When calling reset_root_states() of simulator, the input quaternion is in gym format [x,y,z,w]
        #  simulators will convert it to compatible format if needed.
        rot: List[float] = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat], quaternion sequence definitions are different in gym(xyzw) and genesis/isaaclab(wxyz)
        lin_vel: List[float] = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel: List[float] = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # initial state randomization
        roll_random_scale: float = 0.0
        pitch_random_scale: float = 0.0
        yaw_random_scale: float = 0.0
        default_joint_angles: Dict[str, float] = { # target angles when action = 0.0
            "joint_a": 0.0, 
            "joint_b": 0.0}
    
    class control:
        control_type: str = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness: Dict[str, float] = {'joint_a': 10.0, 'joint_b': 15.0}  # [N*m/rad]
        damping: Dict[str, float] = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale: float = 0.5
        dt: float =  0.02 # control frequency 50Hz
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation: int = 4
    
    class asset:
        # Common
        name: str = None
        file: str = "" # path of urdf file for robot, used in IsaacGym and IsaacLab
        xml_file: str = "" # path of xml file for robot, used in Genesis.
        # name of the feet bodies, bodies containing this substring will be considered as feet
        foot_name: str = ""
        key_bodies: List[str] = [] # list of important bodies to be tracked in mimic tasks
        penalize_contacts_on: List[str] = [] # penalize contacts on links containing these substrings
        terminate_after_contacts_on: List[str] = [] # terminate episode after contacts on links containing these substrings
        fix_base_link = False    # fix base link to the world
        obtain_link_contact_states = False # whether to obtain contact states of specific links, the information can be used for privilege policy
        contact_state_link_names = ["thigh", "calf", "foot"]
        base_link_name = "" # full name of the base link
        self_collisions: int = 0 # 1 to disable, 0 to enable...bitwise filter
        dof_names: List[str] = ["joint_a", "joint_b"] # specify the sequence of dofs in the actions and observations
        dof_armature: List[float] = [0.0] # armature of each dof, 
        # For Genesis
        links_to_keep: List[str] = []  # links that are not merged because of fixed joints
        dof_vel_limits: List[float] = [] # rad/s, obtain from urdf
        # For IsaacGym and IsaacLab
        disable_gravity: bool = False
        collapse_fixed_joints: bool = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse=\"true\">"
        default_dof_drive_mode = 3   # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        density: float = 0.001
        angular_damping: float = 0.0
        linear_damping: float = 0.0
        max_angular_velocity: float = 1000.0
        max_linear_velocity: float = 1000.0
        armature: float = 0.0
        thickness: float = 0.01

    class rewards:
        class scales:
            termination: float = -0.0
            tracking_lin_vel: float = 0.0 # 1.0
            tracking_ang_vel: float = 0.0 # 0.5
            lin_vel_z: float = 0.0 # -2.0
            ang_vel_xy: float = 0.0 # -0.05
            orientation: float = -0.0
            torques: float = 0.0 # -0.00001
            dof_vel: float = -0.0
            dof_acc: float = 0.0 # -2.5e-7
            base_height: float = -0.0 
            feet_air_time: float = 0.0 # 1.0
            collision: float = 0.0 # -1.
            feet_stumble: float = -0.0 
            action_rate: float = 0.0 # -0.01
            dof_pos_stand_still: float = -0.0
        
        only_positive_rewards: bool = True
        tracking_sigma: float = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit: float = 1.0 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit: float = 1.0
        soft_torque_limit: float = 1.0
        base_height_target: float = 1.0
        foot_clearance_target: float = 0.04 # desired foot clearance above ground [m]
        foot_height_offset: float = 0.0     # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma: float = 0.01
    
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        curriculum_threshold = 0.8 # threshold for curriculum learning, if the tracking reward is above this threshold, increase the command range
        zero_cmd_prob = 0.4    # probability of sampling zero command when resampling commands, to encourage the robot to learn standing still behavior
        class ranges:
            lin_vel_x: List[float] = [-1.0, 1.0] # min max [m/s]
            lin_vel_y: List[float] = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw: List[float] = [-1.0, 1.0]    # min max [rad/s]
            heading: List[float] = [-3.14, 3.14]
    
    class domain_rand:
        # randomize rigid body friction
        randomize_friction: bool = True
        friction_range: List[float] = [0.5, 1.25]
        # randomize restitution
        randomize_restitution: bool = False
        restitution_range: List[float] = [0.0, 0.5]
        # randomize base link mass
        randomize_base_mass: bool = True
        added_mass_range: List[float] = [-1.0, 1.0]
        # apply random velocity perturbations to the base link
        push_robots: bool = True
        push_interval_s: int = 15
        max_push_vel_xy: float = 1.0
        # randomize the position of Center of Mass (CoM) to simulate modeling errors
        randomize_com_displacement: bool = True
        com_pos_x_range: List[float] = [-0.01, 0.01]
        com_pos_y_range: List[float] = [-0.01, 0.01]
        com_pos_z_range: List[float] = [-0.01, 0.01]
        # apply random delay to the actions to simulate latency in the control loop
        randomize_ctrl_delay: bool = False
        ctrl_delay_step_range: List[int] = [0, 1] # number of control steps
        # randomize PD gains by a scale factor
        randomize_pd_gain: bool = False
        kp_range: List[float] = [0.8, 1.2]
        kd_range: List[float] = [0.8, 1.2]
        # ! Randomizing joint armature/friction/damping in Genesis require batching dofs/links info, 
        # ! which will slow the simulation greatly.
        # ! It is recommended to keep them false. If needed, please use it in IsaacGym and IsaacLab.
        randomize_joint_armature: bool = False
        joint_armature_range: List[float] = [0.0, 0.05]  # [N*m*s/rad]
        randomize_joint_friction: bool = False
        joint_friction_range: List[float] = [0.0, 0.1]
        randomize_joint_damping: bool = False
        joint_damping_range: List[float] = [0.0, 1.0]
        # Apply random push forces to the links of the robot
        push_links: bool = False
        max_push_force: float = 10.0 # [N], maximum magnitude of the random push force applied to each link
        push_links_interval_s: float = 15.0 # time interval between random pushes
        # randomize position of the camera relative to the fixed pos
        randomize_camera_pos: bool = False
        camera_com_displacement_range: List[float] = [0.01, 0.01, 0.01] # [m], random displacement of the camera position along x, y, z axes
        # randomize orientation of the camera
        randomize_camera_euler: bool = False
        camera_euler_range: List[float] = [0.1, 0.1, 0.1] # [rad], randomize roll, pitch, yaw of the camera
        
    class normalization:
        class obs_scales:
            lin_vel: float = 1.0
            ang_vel: float = 0.25
            dof_pos: float = 1.0
            dof_vel: float = 0.05
            height_measurements: float = 2.0
            depth_image: float = 2.0 # scale from [-0.5, 0.5] to [-1.0, 1.0]
            actions: float = 1.0
        clip_observations: float = 100.0
        clip_actions: float = 100.0

    class noise:
        add_noise: bool = True
        noise_level: float = 1.0 # scales other values
        class noise_scales:
            dof_pos: float = 0.01
            dof_vel: float = 0.5
            lin_vel: float = 0.1
            ang_vel: float = 0.2
            gravity: float = 0.05
            height_measurements: float = 0.1
    
    # constraints config for CaT (Constraints as Termination)
    class constraints:
        class limits:
            pass
        
    # viewer camera:
    class viewer:
        ref_env: int = 0
        pos: List[float] = [2.0, 2.0, 1.0]       # [m], relative to the robot position
        lookat: List[float] = [0.0, 0.0, 0.0]  # [m], relative to the robot position
        rendered_envs_idx: List[int] = [i for i in range(5)]  # [Genesis] number of environments to be rendered, if not headless
    
    # sensor configuration:
    class sensor:
        add_depth: bool = False
        class depth_camera_config:
            num_sensors: int = 1         # number of depth cameras, currently only support 1
            num_history: int = 1         # history frames for depth images
            near_clip: float = 0.1         # near clip distance of the depth camera
            far_clip: float = 5.0          # far clip distance of the depth camera
            near_plane: float = 0.1        # near plane of the depth camera
            far_plane: float = 10.0        # far plane of the depth camera
            resolution: Tuple[int, int] = (60, 80)   # (height, width)
            horizontal_fov_deg: int = 75 # horizontal field of view in degrees
            pos: Tuple[float, float, float] = (0.3, 0.0, 0.1)
            euler: Tuple[float, float, float] = (0.0, 1.57, 0.0)     # (roll, pitch, yaw) in radians, for Warp Camera convention
            decimation: int = 5           # decimation steps for depth image update, relative to control step
            # Warp only
            calculate_depth: bool = True
            segmentation_camera: bool = False
            return_pointcloud: bool = False
            pointcloud_in_world_frame: bool = False

    class sim:
        # Common
        dt: float = 0.005                 # 200 Hz
        substeps: int = 1
        # For Genesis
        max_collision_pairs = 100  # More collision pairs will occupy more GPU memory and slow down the simulation
        IK_max_targets = 2         # Fewer IK targets will lead to fewer memory usage
        # For IsaacGym
        gravity: List[float] = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis: int = 1  # 0 is y, 1 is z
        use_gpu_pipeline: bool = True

        # PhysX engine parameters, for IsaacGym only
        class physx:
            use_gpu: bool = True
            num_subscenes: int = 0
            num_threads: int = 10
            solver_type: int = 1  # 0: pgs, 1: tgs
            num_position_iterations: int = 4
            num_velocity_iterations: int = 0
            contact_offset: float = 0.01  # [m]
            rest_offset: float = 0.0   # [m]
            bounce_threshold_velocity: float = 0.5 #0.5 [m/s]
            max_depenetration_velocity: float = 1.0
            max_gpu_contact_pairs: int = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier: int = 5
            contact_collection: int = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
    
class LeggedRobotCfgPPO(BaseConfig):
    seed: int = 1
    runner_class_name: str = 'OnPolicyRunner'
    class policy:
        clip_actions: float = LeggedRobotCfg.normalization.clip_actions
        init_noise_std: float = 1.0
        actor_hidden_dims: List[int] = [512, 256, 128]
        critic_hidden_dims: List[int] = [512, 256, 128]
        activation: str = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef: float = 1.0
        use_clipped_value_loss: bool = True
        clip_param: float = 0.2
        entropy_coef: float = 0.01
        num_learning_epochs: int = 5
        num_mini_batches: int = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate: float = 1e-3 #5e-4
        schedule: str = 'adaptive' # could be adaptive, fixed
        gamma: float = 0.99
        lam: float = 0.95
        desired_kl: float = 0.01
        max_grad_norm: float = 1.0
        
        # Whether to use SPO(Simple Policy Optimization), refer to refer to https://arxiv.org/abs/2401.16025
        # SPO may collapse with default param settings for PPO, especially with high learning rate
        # learning_rate=2.5e-4, schedule='fixed' are validated
        use_spo: bool = False 

    class runner:
        policy_class_name: str = 'ActorCritic'
        algorithm_class_name: str = 'PPO'
        num_steps_per_env: int = 24 # per iteration
        max_iterations: int = 1500 # number of policy updates

        # logging
        sync_wandb: bool = False  # whether to sync log to wandb
        save_interval: int = 50 # check for potential saves every this many iterations
        experiment_name: str = 'test'
        run_name: str = ''
        # load and resume
        resume: bool = False
        load_run: int = -1 # -1 = last run
        checkpoint: int = -1 # -1 = last saved model
        resume_path: str = None # updated from load_run and chkpt
