from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import K1FlatCommonCfg

"""
Booster K1 DeepMimic environment configuration file.
"""
class K1DeepMimicCfg(K1FlatCommonCfg):
    class env(K1FlatCommonCfg.env):
        frame_stack = 5
        ref_motion_frame_stack = 2
        ref_motion_single_obs = 69
        num_single_obs = 88 + int(ref_motion_single_obs * ref_motion_frame_stack)
        num_observations = int(num_single_obs * frame_stack)
        c_frame_stack = 5
        num_single_critic_obs = num_single_obs + 17
        num_privileged_obs = int(num_single_critic_obs * c_frame_stack)
        num_actions = 22
        # reference motion file, should be a .pkl file containing a dictionary
        motion_file = f'booster_k1/{SIMULATOR}_run/C1_-_stand_to_run_stageii_{SIMULATOR}.pkl'
        episode_length_s = 10
        debug_draw_key_body_points = True # draw key body points for mimic tasks
        max_projected_gravity = -0.3
    
    class init_state(K1FlatCommonCfg.init_state):
        reference_state_initialization = True
        reference_state_initialization_prob = 0.7

    class rewards(K1FlatCommonCfg.rewards):
        soft_dof_pos_limit = 0.99
        tracking_dof_pos_sigma = 4.0
        tracking_dof_vel_sigma = 100.0
        tracking_ref_base_pose_sigma = 0.2
        tracking_ref_base_vel_sigma = 1.0
        tracking_ref_key_pos_sigma = 0.1
        only_positive_rewards = False
        class scales(K1FlatCommonCfg.rewards.scales):
            # limits
            dof_pos_limits = -1.0
            # tasks
            tracking_ref_dof_pos = 0.5
            tracking_ref_dof_vel = 0.1
            tracking_ref_base_pose = 0.5
            tracking_ref_base_vel = 0.1
            tracking_ref_key_pos = 0.15
            # regularization
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            dof_power = -1.e-4
            collision = -1.0
            action_rate = -0.01
            action_smoothness = -0.01
            feet_slip = -0.5
    
    class domain_rand(K1FlatCommonCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 0.5
        randomize_com_displacement = True
        com_pos_x_range = [-0.05, 0.05]
        com_pos_y_range = [-0.05, 0.05]
        com_pos_z_range = [-0.05, 0.05]
    
    class normalization(K1FlatCommonCfg.normalization):
        clip_actions = 100.0

class K1DeepMimicCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        clip_actions = K1DeepMimicCfg.normalization.clip_actions
        critic_hidden_dims = [1024, 256, 128]
        activation = 'elu'
        
    class runner(LeggedRobotCfgPPO.runner):
        num_steps_per_env = 32
        max_iterations = 3000
        run_name = f'k1_deepmimic_{SIMULATOR}'
        experiment_name = 'k1_deepmimic'
        save_interval = 500
