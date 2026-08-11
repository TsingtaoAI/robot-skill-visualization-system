# This file contains template configuration classes for legged robot tasks
# These classes serve as base templates for task-specific configurations
# Author: Genesis LR

from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# ----- Template configuration for AMP tasks -----#
from legged_gym import LEGGED_GYM_ROOT_DIR
import glob

MOTION_FILES = glob.glob(LEGGED_GYM_ROOT_DIR + "/resources/reference_motion/*")

class LeggedRobotAMPCfg(LeggedRobotCfg):
    
    class env(LeggedRobotCfg.env):
        amp_motion_files = MOTION_FILES
    
    class init_state(LeggedRobotCfg.init_state):
        # whether to initialize the robot with the reference motion
        reference_state_initialization = True
        reference_state_initialization_prob = 0.7
        

class LeggedRobotAMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'AMPRunner'
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        amp_replay_buffer_size = 1000000
        disc_lr = 1e-4

    class runner( LeggedRobotCfgPPO.runner ):
        algorithm_class_name = 'PPO_AMP'
        
        amp_reward_coef = 2.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_discr_hidden_dims = [1024, 512]
        amp_task_reward_lerp = 0.3                 # Task reward blending ratio


# ----- Template configuration for Teacher-Student framework -----#
class LeggedRobotTSCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_observations = 48
        num_privileged_obs = None
        # for teacher-student framework
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = num_privileged_obs
        c_frame_stack = 5
        num_single_critic_obs = num_observations
        num_critic_obs = c_frame_stack * num_single_critic_obs

class LeggedRobotTSCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'TSRunner'
    class policy( LeggedRobotCfgPPO.policy ):
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_type = "MLP" # "MLP" or "TCN"
        history_encoder_hidden_dims = [256, 128]       # for MLP
        history_encoder_channel_dims = [1, 1, 1, 1]    # for TCN
        history_encoder_dilation = [1, 1, 2, 1]        # for TCN
        history_encoder_stride = [1, 2, 1, 2]          # for TCN
        history_encoder_final_layer_dim = 128          # for TCN
        kernel_size = 5
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # for encoder training
        encoder_lr = 1.e-3
        num_encoder_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticTS'
        algorithm_class_name = 'PPO_TS'


# ----- Template configuration for Teacher-Student with Depth -----#
class LeggedRobotTSDepthCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_camera_envs = 1 # number of envs with depth camera, starting from the first env
        num_observations = 45
        num_privileged_obs = None
        # for teacher-student framework
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
        num_latent_dims = num_privileged_obs
        c_frame_stack = 5
        num_single_critic_obs = num_observations
        num_critic_obs = c_frame_stack * num_single_critic_obs

    class normalization( LeggedRobotCfg.normalization):
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            depth_image = 2.0
            height_measurements = 1.0
        clip_actions = 100.0
    
class LeggedRobotTSDepthCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'TSDepthRunner'
    class policy( LeggedRobotCfgPPO.policy ):
        critic_hidden_dims = [1024, 256, 128]
        actor_hidden_dims = [512, 256, 128]
        privilege_encoder_hidden_dims = [256, 128]
        cnn_input_channel = LeggedRobotTSDepthCfg.sensor.depth_camera_config.num_history
        cnn_channel_dims = [4, 8]
        cnn_strides = [1, 1]
        cnn_fc_layer_dims = [128]
        combination_mlp_dims = [128, 32]
        cnn_kernel_sizes = [5, 3]
        rnn_type = 'gru'
        rnn_hidden_size = 256
        rnn_num_layers = 1
    
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        encoder_lr = 2.e-4
        
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticTSDepth"
        algorithm_class_name = "PPO_TSDepth"


# ----- Template configuration for DreamWaQ -----#
class LeggedRobotDreamwaqCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_observations = 45  # num_obs
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = 16
        num_explicit_dims = 24  # base linear velocity
        num_decoder_output = num_observations
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 31 + 81 + 17 + 3
        num_privileged_obs = c_frame_stack * num_single_critic_obs

class LeggedRobotDreamwaqCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "DreamWaQRunner" # DreamWaQ Runner
    class policy( LeggedRobotCfgPPO.policy ):
        encoder_hidden_dims = [256, 128]
        decoder_hidden_dims = [256, 128]
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # for vae training
        encoder_lr = 2.e-4
        num_encoder_epochs = 1
        vae_kld_weight = 2.0

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticDreamWaQ"
        algorithm_class_name = "PPO_DreamWaQ"


# ----- Template configuration for CaT (Constraints as Termination) -----#
class LeggedRobotCTSCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_observations = 48
        num_privileged_obs = 94
        num_teacher = 1  # number of teacher envs
        # for teacher-student framework
        # Privileged_obs and critic_obs are seperated here
        # privileged_obs contains information given to privileged encoder
        # critic_obs contains information given to critic, including some privileged information
        # This operation is to prevent the critic from receiving noisy input from the concatenation of current observation(noisy) and latent vector
        frame_stack = 20    # number of frames to stack for obs_history
        num_history_obs = int(num_observations * frame_stack)
        num_latent_dims = num_privileged_obs
        c_frame_stack = 5
        num_single_critic_obs = num_observations
        num_critic_obs = c_frame_stack * num_single_critic_obs

class LeggedRobotCTSCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'CTSRunner'
    class policy( LeggedRobotCfgPPO.policy ):
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_hidden_dims = [256, 128]       # for MLP
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # for encoder training
        encoder_lr = 1.e-3
        num_encoder_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticCTS'
        algorithm_class_name = 'PPO_CTS'


# ----- Template configuration for Explicit Estimator -----#
class LeggedRobotEECfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        # Here the privileged_obs is actually critic_obs
        num_single_obs = 45
        frame_stack = 10    # number of frames to stack for obs_history
        num_estimator_features = int(num_single_obs * frame_stack)
        num_estimator_labels = 24
        c_frame_stack = 5
        num_single_critic_obs = num_single_obs + 31 + 81 + 17
        num_privileged_obs = c_frame_stack * num_single_critic_obs

class LeggedRobotEECfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'EERunner' # Explicit Estimator Runner
    class policy( LeggedRobotCfgPPO.policy ):
        estimator_hidden_dims = [256, 128]
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # for estimator training
        estimator_lr = 2.e-4
        num_estimator_epochs = 1

    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = 'ActorCriticEE'
        algorithm_class_name = 'PPO_EE'
