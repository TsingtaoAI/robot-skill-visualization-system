from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import SIMULATOR

# TRON1 Sole Foot
class TRON1SFCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_single_obs = 33
        frame_stack = 10    # Policy frame stack number
        c_frame_stack = 10  # Critic frame stack number
        num_observations = int( num_single_obs * frame_stack )
        num_single_privileged_obs = num_single_obs + 39
        num_privileged_obs = int( num_single_privileged_obs * c_frame_stack )
        num_actions = 8
        env_spacing = 2.0
        max_projected_gravity = -0.4
    
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = "plane" # none, plane, heightfield
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.85]     # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "ankle_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "ankle_R_Joint": 0.0
        }
        # sit mode pose
        sit_pos = [0.0, 0.0, 0.6] # x,y,z [m]
        sit_joint_angles = {  # target angles when action = 0.0
            "abad_L_Joint": 0.0,
            "hip_L_Joint": 0.58,
            "knee_L_Joint": 1.35,
            "ankle_L_Joint": -0.8,
            "abad_R_Joint": 0.0,
            "hip_R_Joint": -0.58,
            "knee_R_Joint": -1.35,
            "ankle_R_Joint": -0.8
        }
        sit_init_percent = 0.5  # probability of resetting env in sit pose

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # control_type = 'P'
        stiffness = {
            "abad_L_Joint": 45,
            "hip_L_Joint": 45,
            "knee_L_Joint": 45,
            "abad_R_Joint": 45,
            "hip_R_Joint": 45,
            "knee_R_Joint": 45,

            "ankle_L_Joint": 45,
            "ankle_R_Joint": 45,
        }  # [N*m/rad]
        damping = {
            "abad_L_Joint": 1.5,
            "hip_L_Joint": 1.5,
            "knee_L_Joint": 1.5,
            "abad_R_Joint": 1.5,
            "hip_R_Joint": 1.5,
            "knee_R_Joint": 1.5,

            "ankle_L_Joint": 0.8,
            "ankle_R_Joint": 0.8,
        }  # [N*m*s/rad]
        action_scale = 0.25 # action scale: target angle = actionScale * action + defaultAngle
        decimation = 4      # decimation: Number of control action updates @ sim DT per policy DT

    class asset( LeggedRobotCfg.asset ):
        # Common
        name = "tron1sf"
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/limx_dynamics/SF_TRON1A/urdf/robot.urdf'
        xml_file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/limx_dynamics/SF_TRON1A/xml/robot.xml'
        foot_name = "ankle"
        penalize_contacts_on = ["knee", "hip", "base", "abad"]
        terminate_after_contacts_on = []
        # For Genesis
        dof_names = [           # align with the real robot
            "abad_L_Joint",
            "hip_L_Joint",
            "knee_L_Joint",
            "ankle_L_Joint",
            "abad_R_Joint",
            "hip_R_Joint",
            "knee_R_Joint",
            "ankle_R_Joint",
        ]
        links_to_keep = []
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.75
        foot_clearance_target = 0.1  # desired foot clearance above ground [m]
        foot_height_offset = 0.055   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        foot_distance_threshold = 0.115
        about_landing_threshold = 0.05
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            # limitation
            keep_balance = 1.0
            dof_pos_limits = -2.0
            collision = -1.0
            feet_distance = -100.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            # smooth
            lin_vel_z = -0.5
            base_height = -4.0
            ang_vel_xy = -0.05
            orientation = -5.0
            dof_power = -2.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            # gait
            feet_air_time = 1.0
            no_fly = 0.4
            foot_clearance = 0.5
            foot_landing_vel = -0.15
            hip_pos_zero_command = -10.0
            foot_flat = 0.3
    
    class commands( LeggedRobotCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.0, 2.0]
        randomize_base_mass = True
        added_mass_range = [-0.5, 1.]
        push_robots = True
        push_interval_s = 10
        max_push_vel_xy = 1.0
        randomize_com_displacement = True
        com_pos_x_range = [-0.03, 0.03]
        com_pos_y_range = [-0.03, 0.03]
        com_pos_z_range = [-0.03, 0.03]
        randomize_pd_gain = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]
        randomize_joint_armature = True
        joint_armature_range = [0.11, 0.13]
        randomize_joint_friction = True
        joint_friction_range = [0.00, 0.01]
        randomize_joint_damping = True
        joint_damping_range = [1.4, 1.45]

class TRON1SFCfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'flat'
        if SIMULATOR == "genesis":
            run_name += "_genesis"
        elif SIMULATOR == "isaacgym":
            run_name += "_isaacgym"
        experiment_name = 'tron1_sf'
        save_interval = 500
        max_iterations = 4000