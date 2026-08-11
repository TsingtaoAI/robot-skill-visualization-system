from legged_gym import *
from legged_gym.envs.base.legged_robot_nav_config import LeggedRobotNavCfg, LeggedRobotNavCfgPPO
from legged_gym.envs.base.common_cfgs import Go2RoughCommonCfg

class GO2NavCfg( LeggedRobotNavCfg ):
    
    class env( LeggedRobotNavCfg.env ):
        num_envs = 4096
        num_observations = 49 + 187
        num_privileged_obs = 68 + 187
        num_actions = 12
        env_spacing = 2.0
        episode_length_s = 8.0

    class terrain( Go2RoughCommonCfg.terrain):
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 17x11 = 187 points
        # terrain types: [smooth slope, random uniform, stairs up, stairs down, discrete, stepping stones, gap, pit]
        terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]

    class init_state( Go2RoughCommonCfg.init_state ):
        roll_random_scale = 0.1
        pitch_random_scale = 0.1
        yaw_random_scale = 3.14

    class control( Go2RoughCommonCfg.control ):
        pass

    class asset( Go2RoughCommonCfg.asset ):
        pass

    class rewards( LeggedRobotNavCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.35
        tracking_duration_pos_s = 4.0         # duration for tracking pos rewards active
        tracking_duration_orientation_s = 6.0 # duration for tracking orientation rewards active
        pos_error_threshold = 2.0 # if the error between target and robot base is lower than this threshold, orientation tracking will be activated
        stall_distance_threshold = 1.0 # [m], min distance to target for stall penalty to be active
        stall_velocity_threshold = 0.1 # [m/s], max xy velocity for stall penalty to be active
        only_positive_rewards = False
        class scales( LeggedRobotNavCfg.rewards.scales ):
            # limitation
            termination = -200.0
            dof_pos_limits = -5.0
            dof_vel_limits = -1.0
            torque_limits = -0.5
            collision = -1.0
            # command tracking
            tracking_target_pos = 10.0
            tracking_target_orientation = 5.0
            # smooth
            base_acc = -4.e-4
            lin_vel_z = -1.0
            dof_power = -4.e-4
            dof_acc = -1.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            # regularization
            feet_stumble = -1.0
            stall = -1.0
            stand_still = -1.0
            move_in_direction = 0.5
    
    class commands( LeggedRobotNavCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4      # default: position of the target point(x, y) in base frame, heading, time left to target
        default_pos_z = 0.35  # base height relative to the ground
        class ranges( LeggedRobotNavCfg.commands.ranges ):
            pos_x = [0.0, 1.0]      # m, relative to the terrain's border
            pos_y = [-3.0, 3.0]     # m, relative to the terrain's origin
            heading = [-3.14, 3.14] # rad

    class domain_rand( LeggedRobotNavCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_com_displacement = True
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
    
    class normalization( LeggedRobotNavCfg.normalization ):
        class obs_scales( LeggedRobotNavCfg.normalization.obs_scales ):
            lin_vel = 1.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            base_pos = 0.2
            base_orientation = 1/3.15
            time_to_target = 0.25

class GO2NavCfgPPO( LeggedRobotNavCfgPPO ):
    class runner( LeggedRobotNavCfgPPO.runner ):
        num_steps_per_env = 48
        policy_class_name = 'ActorCritic'
        run_name = ''
        experiment_name = 'go2_nav'
        save_interval = 500
        load_run = "Jan28_23-15-15_"
        checkpoint = -1
        max_iterations = 6000