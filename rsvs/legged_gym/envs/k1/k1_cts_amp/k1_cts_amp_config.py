from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.common_cfgs import K1FlatCommonCfg
from legged_gym.envs.k1.k1_amp.k1_amp_config import K1AMPCfg, K1AMPCfgPPO
from legged_gym import SIMULATOR
from rsl_rl.utils.symmetry_cts import compute_symmetric_states_k1

import glob

MOTION_FILES = glob.glob(LEGGED_GYM_ROOT_DIR + f"/resources/reference_motion/booster_k1/{SIMULATOR}_run/*")

class K1_CTS_AMPCfg(K1FlatCommonCfg):
    class env(K1FlatCommonCfg.env):
        num_envs = 4096
        num_teacher = int(num_envs // 4 * 3)
        frame_stack = 5
        num_observations = 75
        num_history_obs = int(num_observations * frame_stack)
        c_frame_stack = 5
        num_single_critic_obs = num_observations + 76
        num_critic_obs = int(c_frame_stack * num_single_critic_obs)
        num_privileged_obs = 70
        num_latent_dims = 70
        
        num_actions = 22
        amp_motion_files = MOTION_FILES
    
    class init_state(K1AMPCfg.init_state):
        pass
    class rewards(K1AMPCfg.rewards):
        pass
        class scales(K1AMPCfg.rewards.scales):
            tracking_ang_vel = 1.0
    
    class domain_rand(K1AMPCfg.domain_rand):
        pass
    
    class commands(K1AMPCfg.commands):
        pass

class K1_CTS_AMPCfgPPO(K1AMPCfgPPO):
    runner_class_name = 'CTS_AMP_Runner'
    class policy( K1AMPCfgPPO.policy ):
        critic_hidden_dims = [1024, 512, 256]
        privilege_encoder_hidden_dims = [256, 128]
        history_encoder_hidden_dims = [256, 128]       # for MLP
    class algorithm(K1AMPCfgPPO.algorithm):
        # for encoder training
        encoder_lr = 4.e-4
        num_encoder_epochs = 1
        # Symmetry loss config
        symmetry_cfg = {
            "use_data_augmentation" : True,
            "data_augmentation_func": compute_symmetric_states_k1,
            "use_mirror_loss": True,
            "mirror_loss_coeff": 0.1,
        }
    class runner( K1AMPCfgPPO.runner ):
        policy_class_name = 'ActorCriticCTS'
        algorithm_class_name = 'PPO_CTS_AMP'
        
        run_name = f'k1_cts_amp_{SIMULATOR}'