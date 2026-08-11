from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import Go2FlatCommonCfg, get_simulator_suffix

class GO2Cfg( Go2FlatCommonCfg ):
    
    class env( Go2FlatCommonCfg.env ):
        num_envs = 4096
        num_observations = 45 # 48 for only sim, 45 for deployment
        num_privileged_obs = None
        num_actions = 12
        
    class rewards( Go2FlatCommonCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.36
        foot_clearance_target = 0.05 # desired foot clearance above ground [m]
        foot_height_offset = 0.022   # height of the foot coordinate origin above ground [m]
        foot_clearance_tracking_sigma = 0.01
        only_positive_rewards = True
        class scales( Go2FlatCommonCfg.rewards.scales ):
            # limitation
            dof_pos_limits = -1.0
            collision = -1.0
            # command tracking
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # smooth
            lin_vel_z = -0.5
            base_height = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_vel = -5.e-4
            dof_acc = -2.e-7
            action_rate = -0.01
            action_smoothness = -0.01
            torques = -2.e-4
            # gait
            feet_air_time = 1.0
            foot_clearance = 0.5
    
    class commands( Go2FlatCommonCfg.commands ):
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges( Go2FlatCommonCfg.commands.ranges ):
            lin_vel_x = [-0.5, 0.5] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class domain_rand( Go2FlatCommonCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        randomize_com_displacement = True
        com_pos_x_range = [-0.01, 0.01]
        com_pos_y_range = [-0.01, 0.01]
        com_pos_z_range = [-0.01, 0.01]

class GO2CfgPPO( LeggedRobotCfgPPO ):
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'simple_rl' + get_simulator_suffix()
        experiment_name = 'go2'
        save_interval = 200
        max_iterations = 1500