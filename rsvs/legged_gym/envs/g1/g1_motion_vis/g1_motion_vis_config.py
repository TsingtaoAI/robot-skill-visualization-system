from legged_gym import *
from legged_gym.envs.base.common_cfgs import G1Flat29DofCommonCfg

"""
Booster G1 Motion Visualization environment configuration file.
"""
class G1MotionVisCfg(G1Flat29DofCommonCfg):
    class env(G1Flat29DofCommonCfg.env):
        num_observations = 96
        num_privileged_obs = num_observations + 3
        num_actions = 29
        episode_length_s = 10
        debug_draw_key_body_points = True # draw key body points for mimic tasks
