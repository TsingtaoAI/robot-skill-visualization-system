from .base_config import BaseConfig

class LeggedRobotNavCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 50
        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10 # episode length in seconds
        debug = False # if debugging, visualize contacts, etc.
        env_spacing = 1.0
        fail_to_terminal_time_s = 0.1 # time before a fail state leads to environment reset, refer to https://github.com/limxdynamics/tron1-rl-isaacgym/tree/master
        debug_draw_height_points_around_base = False # obtain height measurements around the base
        debug_draw_height_points_around_feet = False # obtain height measurements around the feet (9 points around each foot, see terrain.measured_points_x/y)
        debug_draw_terrain_height_points = False # draw all height points of the terrain
        debug_draw_key_body_points = False # draw key body points for mimic tasks
        max_projected_gravity = -0.1 # max allowed projected gravity in z axis
        
        
    class terrain:
        
        # heightfield uses a grid of height samples to represent the terrain, creating enormous points
        # trimesh creates terrain mesh directly, reducing the number of triangles compared with heightfield
        mesh_type = 'plane'    # "heightfield" # none, plane, heightfield
        plane_length = 200.0   # [m]. plane size is 200x200x10 by default
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        border_height = 1.0 # [m] height of the border surrounding the terrain
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        # obtain terrain height information around feet (default: 9 points around feet), measure_
        # x  x   x
        # x F(x) x
        # x  x   x (x: height point, F: foot position)
        obtain_terrain_info_around_feet = False
        measure_heights = False
        # positions of the sampling height around the base (relative to the base of the robot)
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 1 # starting curriculum level
        terrain_length = 6.0
        terrain_width = 6.0
        platform_size = 3.0
        num_rows = 4  # number of terrain rows (levels)
        num_cols = 4  # number of terrain cols (types)
        num_subterrains = num_rows * num_cols
        # terrain types: [smooth slope, random uniform, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class init_state:
        pos = [0.0, 0.0, 1.]           # x,y,z [m]
        # [Convention] When calling reset_root_states() of simulator, the input quaternion is in gym format [x,y,z,w]
        #  simulators will convert it to compatible format if needed.
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat], quaternion sequence definitions are different in gym and genesis
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        # initial state randomization
        roll_random_scale = 0.0    # [rad]
        pitch_random_scale = 0.0   # [rad]
        yaw_random_scale = 0.0     # [rad]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        dt =  0.02 # control frequency 50Hz
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset:
        # Common
        name = None
        file = ""
        foot_name = "None"     # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        key_bodies = [] # list of important bodies to be tracked in mimic tasks
        fix_base_link = False    # fix base link to the world
        obtain_link_contact_states = False
        contact_state_link_names = ["thigh", "calf", "foot"]
        self_collisions = 0      # 1 to disable, 0 to enable...bitwise filter
        dof_names = ["joint_a", "joint_b"] # specify the sequence of dofs in the actions and observations
        dof_armature = [0.0] # armature of each dof
        # For Genesis
        links_to_keep = []          # links that are not merged because of fixed joints
        dof_vel_limits = [20.0, 20.0] # [rad/s], corresponds to dof_names order
        # For IsaacGym and IsaacLab
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        default_dof_drive_mode = 3   # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = False # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class rewards:
        class scales:
            termination = -0.0
            tracking_target_pos = 0.0
            tracking_target_orientation = 0.0
            lin_vel_z = 0 # -2.0
            ang_vel_xy = 0 # -0.05
            orientation = -0.
            torques = 0 # -0.00001
            dof_vel = -0.
            dof_acc = 0 # -2.5e-7
            base_height = -0. 
            feet_air_time = 0 # 1.0
            collision = 0 # -1.
            feet_stumble = -0.0 
            action_rate = 0 # -0.01
            stand_still = -0.
        
        only_positive_rewards = True
        tracking_duration_pos_s = 4.0         # duration for tracking pos rewards active
        tracking_duration_orientation_s = 4.0 # duration for tracking orientation rewards active
        stall_distance_threshold = 0.5 # [m], min distance to target for stall penalty to be active
        stall_velocity_threshold = 0.1 # [m/s], max xy velocity for stall penalty to be active
        pos_error_threshold = 2.0 # if the error between target and robot base is lower than this threshold, orientation tracking will be activated
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        foot_clearance_target = 0.04 # desired foot clearance above ground [m]
        foot_height_offset = 0.0     # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4      # default: position of the target point(x, y) in base frame, heading, time left to target
        default_pos_z = 0.34  # base height relative to the ground
        class ranges:
            pos_x = [-6.0, 6.0]     # m, relative to the robot's base
            pos_y = [-6.0, 6.0]     # m
            pos_z = [-0.06, 0.01]   # m
            heading = [-3.14, 3.14] # rad
    
    class domain_rand:
        # randomize rigid body friction
        randomize_friction = True
        friction_range = [0.5, 1.25]
        # randomize restitution
        randomize_restitution = False
        restitution_range = [0.0, 0.5]
        # randomize base link mass
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        # apply random velocity perturbations to the base link
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        # randomize the position of Center of Mass (CoM) to simulate modeling errors
        randomize_com_displacement = True
        com_pos_x_range = [-0.01, 0.01]
        com_pos_y_range = [-0.01, 0.01]
        com_pos_z_range = [-0.01, 0.01]
        # apply random delay to the actions to simulate latency in the control loop
        randomize_ctrl_delay = False
        ctrl_delay_step_range = [0, 1] # number of control steps
        # randomize PD gains by a scale factor
        randomize_pd_gain = False
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        # ! Randomizing joint armature/friction/damping in Genesis require batching dofs/links info, 
        # ! which will slow the simulation greatly.
        # ! It is recommended to keep them false. If needed, please use it in IsaacGym and IsaacLab.
        randomize_joint_armature = False
        joint_armature_range = [0.0, 0.05]  # [N*m*s/rad]
        randomize_joint_friction = False
        joint_friction_range = [0.0, 0.1]
        randomize_joint_damping = False
        joint_damping_range = [0.0, 1.0]
        # Apply random push forces to the links of the robot
        push_links = False
        max_push_force = 10.0 # [N], maximum magnitude of the random push force applied to each link
        push_links_interval_s = 15.0 # time interval between random pushes


    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            base_pos = 0.2
            base_orientation = 1/3.15
            time_to_target = 0.25
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
    
    class constraints:
        class limits:
            pass
        
    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [2.0, 2.0, 1.0]       # [m]
        lookat = [0., 0, 0.]  # [m]
        rendered_envs_idx = [i for i in range(5)]  # number of environments to be rendered
    
    # sensor configuration:
    class sensor:
        add_depth = False
        use_warp = False       # whether to use warp-based model
        class depth_camera_config:
            num_sensors = 1
            num_history = 1        # history frames for depth images
            
            near_clip = 0.1
            far_clip = 10.0
            near_plane = 0.1
            far_plane = 10.0
            resolution = (60, 80)
            horizontal_fov_deg = 75
            pos =   (0.3, 0.0, 0.1)
            euler = (0.0, 0.0, 0.0)
            decimation = 5
            # Warp only
            calculate_depth = True
            segmentation_camera = False
            return_pointcloud = False
            pointcloud_in_world_frame = False
    class sim:
        # Common
        dt =  0.005
        substeps = 1
        # For Genesis
        max_collision_pairs = 100  # More collision pairs will occupy more GPU memory and slow down the simulation
        IK_max_targets = 2         # Fewer IK targets will lead to fewer memory usage
        # For IsaacGym
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z
        use_gpu_pipeline = True

        # PhysX engine parameters, for IsaacGym only
        class physx:
            use_gpu = True
            num_subscenes = 0
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
    
class LeggedRobotNavCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        clip_actions = LeggedRobotNavCfg.normalization.clip_actions
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        
        # Whether to use SPO(Simple Policy Optimization), refer to refer to https://arxiv.org/abs/2401.16025
        # SPO may collapse with default param settings for PPO, especially with high learning rate
        # learning_rate=2.5e-4, schedule='fixed' are validated
        use_spo = False 

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        sync_wandb = False  # whether to sync log to wandb
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt