from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import K1FlatCommonCfg, get_simulator_suffix


class K1Cfg(K1FlatCommonCfg):
    class env(K1FlatCommonCfg.env):
        frame_stack = 5
        num_single_obs = 75
        num_observations = int(num_single_obs * frame_stack)
        c_frame_stack = 5
        num_single_critic_obs = num_single_obs + 3 + 51 + 4 + 1
        num_privileged_obs = int(num_single_critic_obs * c_frame_stack)
        num_actions = 22

    class rewards(K1FlatCommonCfg.rewards):
        soft_dof_pos_limit = 0.99
        base_height_target = 0.54
        foot_height_offset = 0.038
        foot_clearance_target = 0.08
        foot_distance_threshold = 0.12
        about_landing_threshold = 0.04

        only_positive_rewards = False
        class scales(K1FlatCommonCfg.rewards.scales):
            dof_pos_limits = -1.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            feet_distance = -100.0
            keep_balance = 1.0
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -3.0
            base_height = -2.0
            dof_acc = -2.5e-7
            dof_power = -1.e-4
            collision = -1.0
            action_rate = -0.01
            dof_close_to_default = -0.1
            # dof_close_to_default_stand_still = -0.5
            foot_clearance = 0.4
            foot_flat = 0.2
            biped_periodic_gait = 1.0
            # feet_contact_stand_still = 0.5
            foot_landing_vel = -0.15
        
        class periodic_reward_framework:
            '''Periodic reward framework in OSU's paper(https://arxiv.org/abs/2011.01387)'''
            gait_function_type = "step" # can be "step" or "smooth"
            kappa = 20
            # start of swing(a_swing) is all the same
            b_swing = 0.5
            # phase offset of left and right legs
            theta_left = 0.0
            theta_right = 0.5
            gait_period = 0.8  # [s]
    
    class domain_rand(K1FlatCommonCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.4, 1.6]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.5
        randomize_com_displacement = True
        com_pos_x_range = [-0.02, 0.02]
        com_pos_y_range = [-0.02, 0.02]
        com_pos_z_range = [-0.02, 0.02]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_ctrl_delay = True
        ctrl_delay_step_range = [0, 1]
    
    class commands(K1FlatCommonCfg.commands):
        curriculum = True
        max_curriculum = 1.
        resampling_time = 5.0
        heading_command = False
        zero_cmd_prob = 0.3
        class ranges:
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
    
    class noise(K1FlatCommonCfg.noise):
        class noise_scales(K1FlatCommonCfg.noise.noise_scales):
            dof_pos = 0.08

class K1CfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic"
        max_iterations = 5000
        save_interval = 100
        run_name = '' + get_simulator_suffix()
        experiment_name = 'k1_flat'
