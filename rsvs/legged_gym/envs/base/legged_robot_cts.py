from legged_gym.envs.base.legged_robot_ts import *

class LeggedRobotCTS(LeggedRobotTS):
    
    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)
        self.num_teacher = self.cfg.env.num_teacher