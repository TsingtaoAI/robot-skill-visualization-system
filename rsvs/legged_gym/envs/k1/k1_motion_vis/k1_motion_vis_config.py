from legged_gym import *
from legged_gym.envs.base.common_cfgs import K1FlatCommonCfg

"""
Booster K1 Motion Visualization environment configuration file.
"""
class K1MotionVisCfg(K1FlatCommonCfg):
    class env(K1FlatCommonCfg.env):
        num_observations = 75
        num_privileged_obs = num_observations + 3
        num_actions = 22
        episode_length_s = 10
        debug_draw_key_body_points = True # draw key body points for mimic tasks
