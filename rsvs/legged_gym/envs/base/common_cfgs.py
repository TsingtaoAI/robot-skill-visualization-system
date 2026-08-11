# This file contains common configuration classes for robots supported by LeggedGym-Ex
# These common configuration classes are used to define default values for various configuration parameters, which can be overridden by task-specific configuration classes.
# Author: Yasen Jia
# Date: 2026-02-14

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym import SIMULATOR

# =============================================================================
# CONFIG MIXINS - Reusable configuration blocks
# =============================================================================
# These mixin classes can be combined to create robot-specific configs.
# Mixins provide common patterns that reduce duplication across robot variants.

def get_simulator_suffix():
    """Helper function to get simulator suffix for run names."""
    if SIMULATOR == "genesis":
        return '_genesis'
    elif SIMULATOR == "isaacgym":
        return '_isaacgym'
    elif SIMULATOR == "isaaclab":
        return '_isaaclab'
    return ''

#----- Common configuration for Unitree Go2 on flat terrain -----#
class Go2FlatCommonCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.4] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8, # [rad]
            'RL_thigh_joint': 0.8, # [rad]
            'FR_thigh_joint': 0.8, # [rad]
            'RR_thigh_joint': 0.8, # [rad]

            'FL_calf_joint': -1.5, # [rad]
            'RL_calf_joint': -1.5, # [rad]
            'FR_calf_joint': -1.5, # [rad]
            'RR_calf_joint': -1.5, # [rad]
        }
    
    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {'joint': 20.}   # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        dt = 0.02  # control frequency 50Hz
        decimation = 4 # decimation: Number of control action updates @ sim DT per policy DT
        
    class asset(LeggedRobotCfg.asset):
        # Common
        name = "go2"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/unitree_robotics/go2/urdf/go2.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/unitree_robotics/go2/go2.xml'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "Head"]
        # full name of the base link
        base_link_name = "base"
        dof_names = [           # align with the real robot
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint"
        ]
        dof_armature = [0.01] * 12 # use a small armature for go2
        # For Genesis
        links_to_keep = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
        dof_vel_limits = [30.1, 30.1, 15.7, 
                          30.1, 30.1, 15.7, 
                          30.1, 30.1, 15.7, 
                          30.1, 30.1, 15.7]

#----- Common configuration for Unitree Go2 on rough terrain -----#
class Go2RoughCommonCfg(Go2FlatCommonCfg):
    class terrain( LeggedRobotCfg.terrain ):
        if SIMULATOR in ["isaacgym", "isaaclab"]:
            mesh_type = "trimesh"
        else:
            mesh_type = "heightfield"
        border_size = 20.0 # [m]
        curriculum = True
        # rough terrain only:
        obtain_terrain_info_around_feet = True
        measure_heights = True
        measured_points_x = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4] # 9x9=81
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4]
        terrain_length = 8.0
        terrain_width = 8.0
        platform_size = 4.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 10  # number of terrain cols (types)
        # terrain types: [smooth slope, random uniform, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.1, 0.25, 0.25, 0.2]
        
    class init_state( Go2FlatCommonCfg.init_state ):
        pass
    class control( Go2FlatCommonCfg.control ):
        pass
    class asset( Go2FlatCommonCfg.asset ):
        obtain_link_contact_states = True
        contact_state_link_names = ["thigh", "calf", "foot", "base", "hip"]
        penalize_contacts_on = ["thigh", "calf", "base", "Head", "hip"]
        terminate_after_contacts_on = []
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        foot_clearance_target = 0.09 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = True
        class scales( LeggedRobotCfg.rewards.scales ):
            # limitation
            dof_pos_limits = -2.0
            collision = -1.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # smooth
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_power = -2.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            # gait
            feet_air_time = 1.0
            foot_clearance = 0.2
            feet_contact_stand_still = 0.5
            dof_close_to_default_stand_still = -0.5


#----- Common configuration for Unitree G1 on flat terrain (12DOF) -----#
class G1Flat12DofCommonCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]  # x,y,z [m]
        default_joint_angles = {
            'left_hip_yaw_joint': 0.,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.1,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0,
            'right_hip_yaw_joint': 0.,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.1,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'hip_yaw': 100, 'hip_roll': 100, 'hip_pitch': 100, 'knee': 150, 'ankle': 40}
        damping = {'hip_yaw': 2, 'hip_roll': 2, 'hip_pitch': 2, 'knee': 4, 'ankle': 2}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        name = "g1"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/unitree_robotics/g1_description/g1_12dof.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/unitree_robotics/g1_description/g1_12dof.xml'
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        base_link_name = "pelvis"
        flip_visual_attachments = False
        dof_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ]
        dof_armature = [ # same sequence as dof_names
            # refer to https://github.com/unitreerobotics/unitree_rl_lab/tree/main/source/unitree_rl_lab/unitree_rl_lab/assets/robots
            0.010177520, 0.025101925, 0.010177520,
            0.025101925, 2*0.003609725, 2*0.003609725,
            0.010177520, 0.025101925, 0.010177520,
            0.025101925, 2*0.003609725, 2*0.003609725
        ]


#----- Common configuration for Unitree G1 flat ground (29DOF) -----#
class G1Flat29DofCommonCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.8]  # x,y,z [m]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.1,
            'left_hip_roll_joint': 0,
            'left_hip_yaw_joint': 0.,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0,
            'right_hip_pitch_joint': -0.1,
            'right_hip_roll_joint': 0,
            'right_hip_yaw_joint': 0.,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0,
            'waist_yaw_joint': 0.,
            'waist_roll_joint': 0,
            'waist_pitch_joint': 0,
            'left_shoulder_pitch_joint': 0.3,
            'left_shoulder_roll_joint': 0.25,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.97,
            'left_wrist_roll_joint': 0.15,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'right_shoulder_pitch_joint': 0.3,
            'right_shoulder_roll_joint': -0.25,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.97,
            'right_wrist_roll_joint': -0.15,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'hip': 100., 
                     'knee': 150., 
                     'ankle': 40.,
                     'waist_yaw': 200.,
                     'waist_roll': 40.,
                     'waist_pitch': 40.,
                     'shoulder': 40.,
                     'elbow': 40.,
                     'wrist': 40.,
                     }
        damping = {'hip': 2.0,  
                   'knee': 4.0, 
                   'ankle': 2.0,
                   'waist_yaw': 5.0,
                   'waist_roll': 5.0,
                   'waist_pitch': 5.0,
                   'shoulder': 1.0,
                   'elbow': 1.0,
                   'wrist': 1.0,
                   }
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        name = "g1"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/unitree_robotics/g1_description/g1_29dof.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/unitree_robotics/g1_description/g1_29dof.xml'
        foot_name = "ankle_roll"
        key_bodies = ["ankle_roll", "knee", "hip", 
                      "torso", "wrist_yaw", "shoulder_roll",
                      "shoulder_pitch", "elbow"]
        penalize_contacts_on = ["torso", "hip", "knee", "hand", 
                                "shoulder", "elbow", "wrist"]
        terminate_after_contacts_on = ["pelvis"]
        base_link_name = "pelvis"
        dof_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 
            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
        ]
        dof_armature = [ # same sequence as dof_names
            # refer to https://github.com/unitreerobotics/unitree_rl_lab/tree/main/source/unitree_rl_lab/unitree_rl_lab/assets/robots
            0.010177520, 0.025101925, 0.010177520,
            0.025101925, 2*0.003609725, 2*0.003609725,
            0.010177520, 0.025101925, 0.010177520,
            0.025101925, 2*0.003609725, 2*0.003609725,
            0.010177520, 2*0.003609725, 2*0.003609725,
            0.003609725, 0.003609725, 0.003609725,
            0.003609725, 0.003609725, 0.00425, 0.00425, 
            0.003609725, 0.003609725, 0.003609725,
            0.003609725, 0.003609725, 0.00425, 0.00425,
        ]

# ----- Common configuration for Booster K1 on flat terrain (22DOF) -----#
class K1FlatCommonCfg(LeggedRobotCfg):

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.57]  # x,y,z [m]
        default_joint_angles = {
            'AAHead_yaw': 0., "Head_pitch": 0., 
            "ALeft_Shoulder_Pitch": 0., 'Left_Shoulder_Roll': -1.3, "Left_Elbow_Pitch": 0., "Left_Elbow_Yaw": -0.5, 
            "ARight_Shoulder_Pitch": 0., 'Right_Shoulder_Roll': 1.3, "Right_Elbow_Pitch": 0., "Right_Elbow_Yaw": 0.5, 
            "Left_Hip_Pitch": -0.15, "Left_Hip_Roll": 0., "Left_Hip_Yaw": 0., "Left_Knee_Pitch": 0.3, "Left_Ankle_Pitch": -0.15, "Left_Ankle_Roll": 0.,
            "Right_Hip_Pitch": -0.15, "Right_Hip_Roll": 0., "Right_Hip_Yaw": 0., "Right_Knee_Pitch": 0.3, "Right_Ankle_Pitch": -0.15, "Right_Ankle_Roll": 0.
        }

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {
            'Hip': 80., 'Knee_Pitch': 80., 'Ankle': 30.0, 
            'Shoulder': 4.0, 'Elbow': 4.0, 'Head': 4.0}
        damping = {
            'Hip': 2.0, 'Knee_Pitch': 2.0, 'Ankle': 2.0,
            'Shoulder': 1.0, 'Elbow': 1.0, 'Head': 1.0}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        name = "k1"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/booster_robotics/K1/K1_22dof.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/booster_robotics/K1/K1_22dof.xml'
        foot_name = "foot"
        penalize_contacts_on = ["Trunk", "Shank", "Hip", "Arm", "Head"]
        terminate_after_contacts_on = []
        key_bodies = ["Head_2",
                      "left_hand_link", "right_hand_link",
                      "left_foot_link", "right_foot_link"]
        base_link_name = "Trunk"
        dof_names = [
            "AAHead_yaw", "Head_pitch", 
            "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw", 
            "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw", 
            "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
            "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll"
        ]
        dof_armature = [ # same sequence as dof_names
            # refer to https://github.com/BoosterRobotics/booster_train/blob/main/source/booster_train/booster_train/assets/robots/booster.py
            0.001, 0.001, 
            0.001, 0.001, 0.001, 0.001, 
            0.001, 0.001, 0.001, 0.001, 
            0.0478125, 0.0339552, 0.0282528, 0.095625, 0.0282528*2, 0.0282528*2,
            0.0478125, 0.0339552, 0.0282528, 0.095625, 0.0282528*2, 0.0282528*2
        ]