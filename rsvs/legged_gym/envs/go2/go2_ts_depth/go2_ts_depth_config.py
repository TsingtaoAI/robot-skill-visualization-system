from legged_gym import *
from legged_gym.envs.base.template_cfgs import LeggedRobotTSDepthCfgPPO
from legged_gym.envs.base.common_cfgs import Go2RoughCommonCfg, get_simulator_suffix

class Go2TSDepthCfg( Go2RoughCommonCfg ):
    class env( Go2RoughCommonCfg.env ):
        num_envs = 4000
        num_camera_envs = 1500
        num_observations = 45
        num_privileged_obs = 268
        num_latent_dims = 32
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 31 + 15 + 3 + 48 + 171
        num_critic_obs = c_frame_stack * num_single_critic_obs
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
        num_actions = 12
        debug_draw_depth_images = True
        debug_draw_height_points_around_base = False
        debug_draw_terrain_edge_points = False
        debug = False
    
    class terrain( Go2RoughCommonCfg.terrain ):
        # rough terrain only:
        simplify_mesh: bool = True
        measure_heights = True
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8] # 19x9=171
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        terrain_length = 12.0
        terrain_width = 8.0
        platform_size = 4.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [slope, random uniform, stairs up, stairs down, discrete,
        #                 stepping stones, gaps, pit, multiple high platforms, high platform gaps]
        terrain_proportions = [0.1, 0.0, 0.1, 0.1, 0.1,
                               0.1, 0.2, 0.1, 0.1, 0.1]
        terrain_curriculum_difficulty = {
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
            "pit_depth": "0.1 + 0.5 * difficulty",
            "high_platform_params": {
                "high_platform_height": "0.1 + 0.5 * difficulty",
                "high_platform_length": "np.random.uniform(0.6, 1.6)",
                "high_platform_width": "np.random.uniform(1.0, 2.0)",
                "high_platform_interval": "np.random.uniform(1.0, 2.0)"
            },
            "high_platform_gaps_params": {
                "high_platform_height": "0.1 + 0.5 * difficulty",
                "high_platform_length": "np.random.uniform(1.6, 2.0)",
                "high_platform_width": "np.random.uniform(1.0, 2.0)",
                "high_platform_distance_y": "np.random.uniform(0.2, 2.0)",
                "gap_size": "0.1 + difficulty * 0.8"
            }
        }
        
    class init_state( Go2RoughCommonCfg.init_state ):
        roll_random_scale: float = 0.1  # small random roll to make the policy learn to balance in roll direction
        pitch_random_scale: float = 0.1 # small random pitch to make the policy learn to balance in pitch direction and step up/down small obstacles
        yaw_random_scale: float = 3.14  # initialize robot with random yaw to make it learn to rotate
        
    class control( Go2RoughCommonCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 30.}   # [N*m/rad]
        damping = {'joint': 0.75}     # [N*m*s/rad]

    class asset( Go2RoughCommonCfg.asset ):
        # Common: 
        contact_state_link_names = ["base", "Head", "thigh", "calf", "foot"]
        penalize_contacts_on = ["thigh", "calf", "base", "Head"]
        terminate_after_contacts_on = []

    class rewards( Go2RoughCommonCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.4
        foot_clearance_target = 0.08 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        tracking_sigma = 0.2
        only_positive_rewards = True
        feet_edge_threshold = 0.05 # distance threshold below which foot is considered to be near the edge of a terrain
        class scales:
            # limitation
            dof_pos_limits = -2.0
            collision = -10.0
            # command tracking
            tracking_lin_vel = 1.5
            tracking_ang_vel = 1.0
            # smooth
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_power = -2.e-5
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            # gait
            hip_pos = -0.15
            foot_clearance = 0.2
            feet_stumble = -1.0
            # feet_contact_stand_still = 0.1
            feet_near_edge = -1.0

    class commands( Go2RoughCommonCfg.commands ):
        curriculum = True
        max_curriculum = 1.5
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        zero_cmd_prob = 0.1
        class ranges( Go2RoughCommonCfg.commands.ranges ):
            lin_vel_x = [0.0, 0.5] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [0.0, 0.0]     # only give walking forward (+X direction) command, because the depth image has limited FOV
            
    class domain_rand(Go2RoughCommonCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.7]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        push_robots = True
        push_interval_s = 3
        max_push_vel_xy = 0.5
        randomize_com_displacement = True
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        # randomize position of the camera relative to the fixed pos
        randomize_camera_pos = True
        camera_com_displacement_range = [0.01, 0.01, 0.01] # [m], random displacement of the camera position along x, y, z axes
        # randomize orientation of the camera
        randomize_camera_euler = True
        camera_euler_range = [0.1, 0.1, 0.1] # [rad], randomize roll, pitch, yaw of the camera
    
    class normalization( Go2RoughCommonCfg.normalization):
        class obs_scales( Go2RoughCommonCfg.normalization.obs_scales ):
            actions = 0.1
        clip_actions = 10.0
      
    class viewer( Go2RoughCommonCfg.viewer ):
      pos = (-0.5, 1.5, 0.5)
      lookat = (0, 0, 0.0)
    
    class sensor( Go2RoughCommonCfg.sensor ):
        add_depth = True
        class depth_camera_config( Go2RoughCommonCfg.sensor.depth_camera_config ):
            num_sensors = 1
            num_history = 1        # history frames for depth images
            near_clip = 0.1
            far_clip = 2.0
            near_plane = 0.01
            far_plane = 10.0
            resolution = (60, 80)  # (height, width)
            horizontal_fov_deg = 75
            pos =   (0.3, 0.0, 0.1)
            euler = (0.0, 1.57 + 0.3, 0.0) # 0.3 rad downward pitch
            decimation = 5                 # depth update decimation (1 means every control step, 5 means every 5 control steps, etc.)
            # Warp only
            calculate_depth = True
            segmentation_camera = False
            return_pointcloud = False
            pointcloud_in_world_frame = False

class Go2TSDepthCfgPPO( LeggedRobotTSDepthCfgPPO ):
    seed = 42
    class policy( LeggedRobotTSDepthCfgPPO.policy ):
        clip_actions = Go2TSDepthCfg.normalization.clip_actions
        critic_hidden_dims = [1024, 256, 128]
        actor_hidden_dims = [512, 256, 128]
        privilege_encoder_hidden_dims = [256, 128]
        cnn_input_channel = Go2TSDepthCfg.sensor.depth_camera_config.num_history
        cnn_channel_dims = [8, 8]
        cnn_strides = [1, 1]
        cnn_fc_layer_dims = [128, 64]
        combination_mlp_dims = [128, 32]
        cnn_kernel_sizes = [5, 3]
        rnn_type = 'gru'
        rnn_hidden_size = 512
        rnn_num_layers = 1
    
    class algorithm( LeggedRobotTSDepthCfgPPO.algorithm ):
        encoder_lr = 2.e-4
        
    class runner( LeggedRobotTSDepthCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_depth'
        save_interval = 500
        load_run = "Mar18_18-01-46_"
        checkpoint = -1
        max_iterations = 10000