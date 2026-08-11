from legged_gym import *
from legged_gym.envs.base.template_cfgs import LeggedRobotDreamwaqCfgPPO
from legged_gym.envs.base.common_cfgs import Go2RoughCommonCfg, get_simulator_suffix

class Go2DreamwaqCfg( Go2RoughCommonCfg ):
    class env( Go2RoughCommonCfg.env ):
        num_camera_envs = 1 # number of envs with depth camera, starting from the first env
        num_envs = 3200
        num_actions = 12
        num_observations = 45  # num_obs
        frame_stack = 5    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = 16
        num_explicit_dims = 24  # base linear velocity
        num_decoder_output = num_observations
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 31 + 81 + 17 + 3
        num_privileged_obs = c_frame_stack * num_single_critic_obs
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
        debug_draw_depth_images = True

    class commands( Go2RoughCommonCfg.commands ):
        curriculum = True
        max_curriculum = 1.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( Go2RoughCommonCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
            
    class domain_rand(Go2RoughCommonCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.7]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.
        randomize_com_displacement = True
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]


class Go2DreamwaqCfgPPO( LeggedRobotDreamwaqCfgPPO ):
    class policy( LeggedRobotDreamwaqCfgPPO.policy ):
        critic_hidden_dims = [1024, 256, 128]
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [256, 128]
    class algorithm( LeggedRobotDreamwaqCfgPPO.algorithm ):
        encoder_lr = 2.e-4
        num_encoder_epochs = 1
        vae_kld_weight = 2.0
    class runner( LeggedRobotDreamwaqCfgPPO.runner ):
        run_name = 'dreamwaq' + get_simulator_suffix()
        experiment_name = 'go2_rough'
        save_interval = 500
        max_iterations = 3000