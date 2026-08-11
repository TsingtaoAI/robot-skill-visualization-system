# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import *
from .base.legged_robot import LeggedRobot
from .base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
# k1
from legged_gym.envs.k1.k1 import K1Robot
from legged_gym.envs.k1.k1_config import K1Cfg, K1CfgPPO
# k1 motion visualization
from legged_gym.envs.k1.k1_motion_vis.k1_motion_vis import K1MotionVis
from legged_gym.envs.k1.k1_motion_vis.k1_motion_vis_config import K1MotionVisCfg
# k1 DeepMimic
from legged_gym.envs.k1.k1_deepmimic.k1_deepmimic import K1DeepMimic
from legged_gym.envs.k1.k1_deepmimic.k1_deepmimic_config import K1DeepMimicCfg, K1DeepMimicCfgPPO
# k1 AMP
from legged_gym.envs.k1.k1_amp.k1_amp import K1AMP
from legged_gym.envs.k1.k1_amp.k1_amp_config import K1AMPCfg, K1AMPCfgPPO
# k1 cts amp
from legged_gym.envs.k1.k1_cts_amp.k1_cts_amp import K1_CTS_AMP
from legged_gym.envs.k1.k1_cts_amp.k1_cts_amp_config import K1_CTS_AMPCfg, K1_CTS_AMPCfgPPO
# g1
from legged_gym.envs.g1.g1 import G1Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
# g1 motion visualization
from legged_gym.envs.g1.g1_motion_vis.g1_motion_vis import G1MotionVis
from legged_gym.envs.g1.g1_motion_vis.g1_motion_vis_config import G1MotionVisCfg
# g1 DeepMimic
from legged_gym.envs.g1.g1_deepmimic.g1_deepmimic import G1DeepMimic
from legged_gym.envs.g1.g1_deepmimic.g1_deepmimic_config import G1DeepMimicCfg, G1DeepMimicCfgPPO
# go2
from legged_gym.envs.go2.go2 import GO2
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO
# go2 walk these ways
from legged_gym.envs.go2.go2_wtw.go2_wtw import GO2WTW
from legged_gym.envs.go2.go2_wtw.go2_wtw_config import GO2WTWCfg, GO2WTWCfgPPO
# go2_ts(teacher-student)
from legged_gym.envs.go2.go2_ts.go2_ts import Go2TS
from legged_gym.envs.go2.go2_ts.go2_ts_config import Go2TSCfg, Go2TSCfgPPO
# go2_ee(explicit estimator)
from legged_gym.envs.go2.go2_ee.go2_ee import Go2EE
from legged_gym.envs.go2.go2_ee.go2_ee_config import Go2EECfg, Go2EECfgPPO
# go2_cts(concurrent teacher-student)
from legged_gym.envs.go2.go2_cts.go2_cts import Go2CTS
from legged_gym.envs.go2.go2_cts.go2_cts_config import Go2CTSCfg, Go2CTSCfgPPO
# go2_dreamwaq
from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq import Go2Dreamwaq
from legged_gym.envs.go2.go2_dreamwaq.go2_dreamwaq_config import Go2DreamwaqCfg, Go2DreamwaqCfgPPO
# go2_cat(constraint-as-termination)
from legged_gym.envs.go2.go2_cat.go2_cat import Go2CaT
from legged_gym.envs.go2.go2_cat.go2_cat_config import Go2CaTCfg, Go2CaTCfgPPO
# go2_ts_depth
from legged_gym.envs.go2.go2_ts_depth.go2_ts_depth import Go2TSDepth
from legged_gym.envs.go2.go2_ts_depth.go2_ts_depth_config import Go2TSDepthCfg, Go2TSDepthCfgPPO
# go2_nav
from legged_gym.envs.go2.go2_nav.go2_nav import GO2Nav
from legged_gym.envs.go2.go2_nav.go2_nav_config import GO2NavCfg, GO2NavCfgPPO

# tron1_pf
from legged_gym.envs.tron1pf.tron1pf import TRON1PF
from legged_gym.envs.tron1pf.tron1pf_config import TRON1PFCfg, TRON1PFCfgPPO
# tron_pf_ee
from legged_gym.envs.tron1pf.tron1pf_ee.tron1pf_ee import TRON1PF_EE
from legged_gym.envs.tron1pf.tron1pf_ee.tron1pf_ee_config import TRON1PF_EECfg, TRON1PF_EECfgPPO
# tron1_sf
from legged_gym.envs.tron1sf.tron1sf import TRON1SF
from legged_gym.envs.tron1sf.tron1sf_config import TRON1SFCfg, TRON1SFCfgPPO
# bipedal_walker
from legged_gym.envs.bipedal_walker.bipedal_walker_config import BipedalWalkerCfg, BipedalWalkerCfgPPO
from legged_gym.envs.bipedal_walker.bipedal_walker import BipedalWalker
# # go2_sysid
# from legged_gym.envs.go2.go2_sysid.go2_sysid import GO2SysID
# from legged_gym.envs.go2.go2_sysid.go2_sysid_config import GO2SysIDCfg


from legged_gym.utils.task_registry import task_registry

task_registry.register("k1", K1Robot, K1Cfg(), K1CfgPPO())
task_registry.register("k1_deepmimic", K1DeepMimic, K1DeepMimicCfg(), K1DeepMimicCfgPPO())
task_registry.register("k1_motion_vis", K1MotionVis, K1MotionVisCfg(), LeggedRobotCfgPPO()) # for motion visualization, not for training
task_registry.register("k1_amp", K1AMP, K1AMPCfg(), K1AMPCfgPPO())
task_registry.register("k1_cts_amp", K1_CTS_AMP, K1_CTS_AMPCfg(), K1_CTS_AMPCfgPPO()) # unvalidated
task_registry.register("g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())
task_registry.register("g1_deepmimic", G1DeepMimic, G1DeepMimicCfg(), G1DeepMimicCfgPPO())
task_registry.register("g1_motion_vis", G1MotionVis, G1MotionVisCfg(), LeggedRobotCfgPPO()) # for motion visualization, not for training
task_registry.register( "go2", GO2, GO2Cfg(), GO2CfgPPO())
task_registry.register( "go2_wtw", GO2WTW, GO2WTWCfg(), GO2WTWCfgPPO())
task_registry.register( "go2_ts", Go2TS, Go2TSCfg(), Go2TSCfgPPO())
task_registry.register( "go2_ee", Go2EE, Go2EECfg(), Go2EECfgPPO())
task_registry.register( "go2_cts", Go2CTS, Go2CTSCfg(), Go2CTSCfgPPO())
task_registry.register( "go2_dreamwaq", Go2Dreamwaq, Go2DreamwaqCfg(), Go2DreamwaqCfgPPO())
task_registry.register( "go2_cat", Go2CaT, Go2CaTCfg(), Go2CaTCfgPPO())
task_registry.register( "go2_ts_depth", Go2TSDepth, Go2TSDepthCfg(), Go2TSDepthCfgPPO()) # unvalidated
task_registry.register( "go2_nav", GO2Nav, GO2NavCfg(), GO2NavCfgPPO())
task_registry.register( "tron1pf", TRON1PF, TRON1PFCfg(), TRON1PFCfgPPO())
task_registry.register( "tron1pf_ee", TRON1PF_EE, TRON1PF_EECfg(), TRON1PF_EECfgPPO())
task_registry.register( "tron1sf", TRON1SF, TRON1SFCfg(), TRON1SFCfgPPO())
# task_registry.register( "go2_sysid", GO2SysID, GO2SysIDCfg(), GO2CfgPPO())
# task_registry.register( "bipedal_walker", BipedalWalker, BipedalWalkerCfg(), BipedalWalkerCfgPPO())