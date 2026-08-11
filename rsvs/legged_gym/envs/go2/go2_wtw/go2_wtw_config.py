from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import Go2FlatCommonCfg, get_simulator_suffix

class GO2WTWCfg(Go2FlatCommonCfg):
    class env(Go2FlatCommonCfg.env):
        num_envs = 4096
        num_actions = 12
        # observation history
        frame_stack = 5   # policy frame stack
        c_frame_stack = 5  # critic frame stack
        num_single_obs = 61
        num_observations = int(num_single_obs * frame_stack)
        single_num_privileged_obs = num_single_obs + 38
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        env_spacing = 1.0
        
    class rewards(Go2FlatCommonCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_tracking_sigma = 0.01
        foot_height_offset = 0.022 # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        euler_tracking_sigma = 0.1
        about_landing_threshold = 0.03
        only_positive_rewards = True
        class scales(Go2FlatCommonCfg.rewards.scales):
            # limitation
            dof_pos_limits = -10.0
            collision = -1.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            tracking_base_height = 0.6
            tracking_orientation = 0.6
            tracking_foot_clearance = 0.9
            quad_periodic_gait = 1.5
            # smooth
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            dof_vel = -5.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -2.e-4
            foot_landing_vel = -0.1
            hip_pos = -1.0
            
        class periodic_reward_framework:
            '''Periodic reward framework in OSU's paper(https://arxiv.org/abs/2011.01387)'''
            gait_function_type = "step" # can be "step" or "smooth"
            kappa = 20
            # start of swing is all the same
            b_swing = 0.5
            # trot, pronk, pace, bound
            theta_fl_list = [0.0, 0.0, 0.5, 0.0]  # front left leg
            theta_fr_list = [0.5, 0.0, 0.0, 0.0]
            theta_rl_list = [0.5, 0.0, 0.5, 0.5]
            theta_rr_list = [0.0, 0.0, 0.0, 0.5]
        
        class behavior_params_range:
            resampling_time = 5.0
            gait_period_range = [0.3, 0.6]
            foot_clearance_target_range = [0.04, 0.12]
            base_height_target_range = [0.2, 0.34]
            pitch_target_range = [-0.3, 0.3]
            
    class commands(Go2FlatCommonCfg.commands):
        curriculum = True
        max_curriculum = 1.
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges(Go2FlatCommonCfg.commands.ranges):
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand(Go2FlatCommonCfg.domain_rand):
        enable = True
        randomize_friction = enable
        friction_range = [0.2, 1.7]
        randomize_base_mass = enable
        added_mass_range = [-1., 1.]
        push_robots = enable
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_com_displacement = enable
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
        randomize_pd_gain = enable
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

class GO2WTWCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'wtw' + get_simulator_suffix()
        experiment_name = 'go2_wtw'
        save_interval = 500
        max_iterations = 5000
