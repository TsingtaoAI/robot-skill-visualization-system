import time

from legged_gym import *
import os

from legged_gym.envs import *
from legged_gym.utils import *
from legged_gym.utils.viser_viewer import create_viser_viewer

import numpy as np
import torch
from legged_gym.scripts.joystick import Joystick
    
def override_configs(env_cfg, args, task_type):
    """Override some environment configuration parameters for testing

    Args:
        env_cfg: environment configuration
        args: command line arguments
        task_type: type of the task
    """
    # override some parameters for testing
    # number of environments
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 16)
    if task_type == "cts" or task_type == "cts_amp": # concurrent teacher-student specific
        env_cfg.env.num_teacher = 1
    elif "depth" in task_type:  # depth specific
        env_cfg.env.num_envs = 1 # for depth observation, only support num_envs=1 for now
        env_cfg.env.num_camera_envs = 1
    env_cfg.viewer.rendered_envs_idx = list(range(env_cfg.env.num_envs))
    # adjust parameters according to terrain type
    if env_cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = 1
        env_cfg.terrain.border_size = 2.0
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.selected = True
        env_cfg.env.debug_draw_terrain_height_points = False
        env_cfg.domain_rand.push_robots = False
        
        
        # random uniform terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.random_uniform_terrain", 
        #                                   "min_height" : -0.05, "max_height": 0.05, 
        #                                   "step":0.005, "downsampled_scale" : 0.2}
        # slope
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_sloped_terrain",
        #                                   "slope": -0.4, "platform_size": 3.0}
        # stairs
        env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pyramid_stairs_terrain",
                                        "step_width": 0.4, "step_height": -0.1, "platform_size": 3.0}
        # discrete obstacles
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.discrete_obstacles_terrain",
        #                                   "max_height": 0.2,
        #                                   "min_size": 1.0,
        #                                   "max_size": 2.0,
        #                                   "num_rects": 20,
        #                                   "platform_size": 3.0}
        # wave terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.wave_terrain", 
        #                                   "amplitude": 0.1, "num_waves": 2}
        # stepping stones
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.stepping_stones_terrain",
        #                                   "stone_length": 1.0, "stone_width": 0.5, "max_height": 0.0,
        #                                   "stone_distance_x": 1.0, "stone_distance_y": 0.3, "platform_size": 3.0}
        # gap terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.gap_terrain", 
        #                                   "gap_size": 0.8, "platform_size": 4.0}
        # pit terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.pit_terrain", 
        #                                   "depth": 0.5, "platform_size": 4.0}
        # multiple pits terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.multiple_high_platforms_terrain",
        #                                   "high_platform_height": 0.5, "high_platform_length": 0.5, "high_platform_width": 2.0,
        #                                   "high_platform_interval": 1.8, "platform_size": 3.0}
        # high_platform_gaps_terrain
        # env_cfg.terrain.terrain_kwargs = {"type": "terrain_utils.high_platform_gaps_terrain",
        #                                   "high_platform_height": 0.5, "high_platform_length": 1.6, "high_platform_width": 1.0,
        #                                   "high_platform_distance_y": 2.0, "gap_size": 0.6, "platform_size": 4.0}
        
        
    env_cfg.env.debug = True
    env_cfg.commands.zero_cmd_prob = 0.0 # for testing, use non-zero commands all the time

    if args.viewer == "viser":
        args.headless = True
        # Viser sliders drive all commands/behavior params; disable periodic resampling.
        env_cfg.commands.resampling_time = 1e9
        if hasattr(env_cfg.rewards, "behavior_params_range"):
            env_cfg.rewards.behavior_params_range.resampling_time = 1e9
    else:
        env_cfg.commands.ranges.lin_vel_x = [0.5, 0.5]
        env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        env_cfg.commands.ranges.ang_vel_yaw = [-1.0, 1.0]
        env_cfg.commands.ranges.heading = [0.0, 0.0]
    
    if args.use_joystick:
        env_cfg.commands.heading_command = False

    if args.headless or args.viewer == "viser":
        # OpenCV imshow needs a display; disable on headless/SSH/viser runs.
        env_cfg.env.debug_draw_depth_images = False

def print_debug_info(env, robot_index):
    """Print debug information while interacting

    Args:
        env: environment object
        robot_index (int): index of the robot to print info for
    """
    # print debug info
    # print("base lin vel: ", env.simulator.base_lin_vel[robot_index, :].cpu().numpy())
    # print("base yaw angle: ", env.simulator.base_euler[robot_index, 2].item())
    # print("base height: ", env.simulator.base_pos[robot_index, 2].cpu().numpy())
    # print("foot_height: ", env.simulator.feet_pos[robot_index, :, 2].cpu().numpy())
    # print(f"knee pitch: {env.simulator.dof_pos[robot_index, [13,19]].cpu().numpy()}")
    # print(f"feet distance: {torch.norm(env.simulator.feet_pos[robot_index, 0, [0, 1]] - env.simulator.feet_pos[robot_index, 1, [0, 1]]).item()}")
    # print(f"actions: {env.simulator.dof_pos[robot_index].cpu().numpy()}")
    # print(f"command: {env.commands[robot_index].cpu().numpy()}")
    # print(f"dr_ctrl_delay: {env.simulator.dr_ctrl_delay[robot_index].item()}")
    pass

def interaction_loop(env, policy, args, task_type, viser_viewer=None):
    """Run interaction loop between environment and policy

    Args:
        env: environment object
        policy : a policy that takes observations and outputs actions
        args: command line arguments
        viser_viewer: optional ViserViewer for web-based visualization
    """
    
    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 2 # which joint is used for logging
    stop_state_log = 300 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
        
    # Get initial observations according to task type
    if task_type == "ts_depth":
        obs_buf, privileged_obs_buf, depth_image, critic_obs = env.get_observations()
    elif task_type == "ts" or task_type == "cat" or task_type == "cts" or task_type == "cts_amp": # teacher-student specific (including AMP)
        obs_buf, privileged_obs_buf, obs_history, critic_obs = env.get_observations()
    elif task_type == "ee":
        estimator_features, _, _ = env.get_observations()
    elif task_type == "dreamwaq":  # dreamwaq
        obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states = env.get_observations()
    else: # vanilla
        obs_buf = env.get_observations()
    
    # Setup joystick if needed
    if args.use_joystick:
        joystick = Joystick(joystick_type=args.joystick_type)
    
    frame_dt = 1 / 60.0 # 30Hz
    # interaction loop
    for i in range(10*int(env.max_episode_length)):
        
        t_start = time.perf_counter()
        # update commands from joystick
        if args.use_joystick:
            joystick.update()
            env.commands[:, 0] = -joystick.ly
            env.commands[:, 1] = -joystick.lx
            env.commands[:, 2] = -joystick.rx
        # update commands from viser GUI sliders
        elif viser_viewer is not None:
            viser_viewer.apply_commands_to_env(env)
        
        # set the viewer camera to follow the first environment by default
        if args.follow_robot:
            pos = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.pos, dtype=np.float32)
            lookat = env.simulator.base_pos[robot_index].cpu().numpy() + np.array(env.cfg.viewer.lookat, dtype=np.float32)
            env.set_viewer_camera(pos, lookat)
            
        # Step the environment according to task type
        if task_type == "ts_depth":
            actions = policy(obs_buf, depth_image)
            obs_buf, privileged_obs_buf, depth_image, critic_obs, rews, dones, infos = env.step(actions.detach())
        elif task_type == "ts" or task_type == "cat" or task_type == "cts":
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos = env.step(actions.detach())
        elif task_type == "ee":
            actions = policy(estimator_features.detach())
            estimator_features, estimator_labels, _, rews, dones, infos = env.step(actions.detach())
        elif task_type == "dreamwaq":
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, explicit_labels, next_states, rews, dones, infos = env.step(actions.detach())
        elif task_type == "amp":
            actions = policy(obs_buf.detach())
            obs_buf, _, rews, dones, infos, _, _ = env.step(actions.detach())
        elif task_type == "cts_amp":
            actions = policy(obs_buf, obs_history)
            obs_buf, privileged_obs_buf, obs_history, critic_obs, rews, dones, infos, _, _ = env.step(actions.detach())
        else:
            actions = policy(obs_buf.detach())
            obs_buf, _, rews, dones, infos = env.step(actions.detach())
        
        if viser_viewer is not None:
            viser_viewer.update_from_simulator(env, robot_index)

        print_debug_info(env, robot_index)
        
        # Update logger info
        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.simulator.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.simulator.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.simulator.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.simulator.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.simulator.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.simulator.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.simulator.base_ang_vel[robot_index, 2].item(),
                    # 'contact_forces_z': env.feet_max_force_z[robot_index, 
                    #                                             env.simulator.feet_contact_indices].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
        
        # sleep for the remainder of the frame budget to match real-time playback
        elapsed = time.perf_counter() - t_start
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

def export_policy(alg_runner, path: str, args, env_cfg, train_cfg, task_type):
    """export the policy as jit script according to different task types

    Args:
        alg_runner: algorithm runner
        path (str): path to which the policy is exported
        args: command line arguments
        env_cfg: environment configuration
        train_cfg: training configuration
    """
    if task_type == "ts_depth":
        exporter = PolicyExporterDepth(alg_runner.alg.actor_critic, train_cfg)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif task_type == "ts" or task_type == "cat" or task_type == "cts" or task_type == "cts_amp":
        exporter = PolicyExporterTS(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif task_type == "ee":
        exporter = PolicyExporterEE(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    elif task_type == "dreamwaq":
        exporter = PolicyExporterWaQ(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    else:
        exporter = PolicyExporter(alg_runner.alg.actor_critic)
        exporter.export(path, env_cfg, args.export_onnx, train_cfg)
    
    print('Exported policy as jit script to: ', path)
    if args.export_onnx:
        print('Exported policy as onnx to: ', path)
    

def play(args):
    """Main function to run the play script

    Args:
        args (_type_): command line arguments
    """
    if SIMULATOR == "genesis":
        gs.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            logging_level='warning',
        )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    splitted = args.task.split("_")
    # by default, the first part of the task name is robot name, and the second part is task type, e.g. go2_ts, go2_cat, go2_ee, go2_cts, go2_dreamwaq, go2_ts_depth
    # concatenate the parts after the first part to get the task type, e.g. ts, cat, ee, cts, dreamwaq, ts_depth
    task_type = "_".join(splitted[1:])
    print("Task type: ", task_type)
    override_configs(env_cfg, args, task_type)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++ or python)
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 
                            train_cfg.runner.load_run, 'exported')
    export_policy(ppo_runner, path, args, env_cfg, train_cfg, task_type)
    
    viser_viewer = None
    if args.viewer == 'viser':
        viser_viewer = create_viser_viewer(env, port=args.viser_port)
        print(f"Viser web viewer started at http://localhost:{args.viser_port}")
    
    interaction_loop(env, policy, args, task_type, viser_viewer=viser_viewer)
    
    if viser_viewer is not None:
        viser_viewer.stop()
    
    
if __name__ == '__main__':
    args = get_args()
    play(args)
