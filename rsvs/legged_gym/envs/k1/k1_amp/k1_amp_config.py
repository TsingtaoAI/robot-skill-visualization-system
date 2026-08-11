from legged_gym.envs.base.template_cfgs import LeggedRobotAMPCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.common_cfgs import K1FlatCommonCfg
from legged_gym import SIMULATOR
from rsl_rl.utils.symmetry import compute_symmetric_states_k1

import glob

MOTION_FILES = glob.glob(LEGGED_GYM_ROOT_DIR + f"/resources/reference_motion/booster_k1/{SIMULATOR}_run/*")

class K1AMPCfg(K1FlatCommonCfg):
    class env(K1FlatCommonCfg.env):
        num_envs = 4096
        frame_stack = 5
        num_single_obs = 75
        num_observations = int(num_single_obs * frame_stack)
        c_frame_stack = 5
        num_single_critic_obs = num_single_obs + 75
        num_privileged_obs = int(num_single_critic_obs * c_frame_stack)
        num_actions = 22
        amp_motion_files = MOTION_FILES
    
    class init_state(K1FlatCommonCfg.init_state):
        # whether to initialize the robot with the reference motion
        reference_state_initialization = True
        reference_state_initialization_prob = 0.0
    
    class rewards(K1FlatCommonCfg.rewards):
        soft_dof_pos_limit = 0.99
        base_height_target = 0.54
        foot_height_offset = 0.038
        foot_clearance_target = 0.10
        foot_distance_threshold = 0.16
        about_landing_threshold = 0.04
        only_positive_rewards = False
        class scales(K1FlatCommonCfg.rewards.scales):
            # task
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.2
            keep_balance = 1.0
            # smooth
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -2.0
            dof_acc = -2.5e-7
            dof_power = -1.e-4
            collision = -1.0
            action_rate = -0.01
            action_smoothness = -0.01
            # regularization
            feet_distance = -100.0
            hip_yaw_roll_pos = -0.2
            arm_pos = -0.01
            feet_slip = -0.5
            foot_clearance = 0.5
            foot_flat = 0.2
            foot_landing_vel = -0.15
            biped_periodic_gait = 1.0
            # feet_contact_stand_still = 0.5
            # dof_close_to_default_stand_still = -0.5
        
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
        friction_range = [0.2, 1.6]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        push_robots = True
        push_interval_s = 15
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
        resampling_time = 10.0
        heading_command = False
        zero_cmd_prob = 0.2
        class ranges:
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-0.4, 0.4]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]
    
    class viewer(K1FlatCommonCfg.viewer):
        pos = [1.0, 1.0, 0.5]
    
    class noise(K1FlatCommonCfg.noise):
        class noise_scales(K1FlatCommonCfg.noise.noise_scales):
            dof_pos = 0.08

class K1AMPCfgPPO(LeggedRobotAMPCfgPPO):
    class policy(LeggedRobotAMPCfgPPO.policy):
        critic_hidden_dims = [1024, 512, 256]
    class algorithm(LeggedRobotAMPCfgPPO.algorithm):
        amp_replay_buffer_size = K1AMPCfg.env.num_envs * LeggedRobotAMPCfgPPO.runner.num_steps_per_env * 10
        disc_lr = 1.e-4
        # Symmetry loss config
        symmetry_cfg = {
            "use_data_augmentation" : True,
            "data_augmentation_func": compute_symmetric_states_k1,
            "use_mirror_loss": True,
            "mirror_loss_coeff": 0.1,
        }
    class runner( LeggedRobotAMPCfgPPO.runner ):
        amp_reward_coef = 1.0 * K1AMPCfg.control.dt # consistent with other rewards by multiplying dt
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = K1AMPCfg.env.num_envs * LeggedRobotAMPCfgPPO.runner.num_steps_per_env * 10
        amp_discr_hidden_dims = [1024, 512]
        amp_task_reward_lerp = 0.5                 # 任务奖励混合比例
        
        max_iterations = 20000
        save_interval = 200
        run_name = f'k1_amp_{SIMULATOR}'
        experiment_name = 'k1_amp'