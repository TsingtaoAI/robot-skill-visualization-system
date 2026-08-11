import os
import copy
import torch
import numpy as np
import random
import argparse

from rsl_rl.modules import ActorCriticTSDepth

try:
    from rich_argparse import RichHelpFormatter
    HAS_RICH_ARGPARSE = True
except ImportError:
    HAS_RICH_ARGPARSE = False

def class_to_dict(obj) -> dict:
    if not hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    """Override some configuration parameters from command line arguments
       Called in task_registry.py/make_env()

    Args:
        env_cfg : environment configuration
        cfg_train : training configuration
        args : command line arguments

    Returns:
        env_cfg : updated environment configuration
        cfg_train : updated training configuration
    """
    # environment parameters
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.debug:
            env_cfg.env.debug = args.debug
        if args.motion_file is not None:
            env_cfg.env.motion_file = args.motion_file
        if args.num_student is not None:
            env_cfg.env.num_camera_envs = args.num_student
            
    # training parameters
    if cfg_train is not None:
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.sync_wandb:
            cfg_train.runner.sync_wandb = args.sync_wandb
        if args.ckpt is not None:
            cfg_train.runner.checkpoint = args.ckpt
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run

    return env_cfg, cfg_train

def get_args():
    """Parse command line arguments

    Returns:
        args: parsed command line arguments
    """
    # Use RichHelpFormatter for colored help if available
    formatter_class = RichHelpFormatter if HAS_RICH_ARGPARSE else argparse.HelpFormatter
    
    parser = argparse.ArgumentParser(
        formatter_class=formatter_class,
        description="LeggedGym-Ex - Train legged robots with reinforcement learning",
        epilog="For more information, visit: https://github.com/lupinjia/LeggedGym-Ex"
    )
    parser.add_argument('--task',           type=str, default='go2', help="task name")
    parser.add_argument('--headless',       action='store_true', default=False, help="enable visualization by default")
    parser.add_argument('--cpu',            action='store_true', default=False, help="use CPU instead of CUDA")
    parser.add_argument('--num_envs',       type=int, default=None, help="number of parallel environments")
    parser.add_argument('--max_iterations', type=int, default=None, help="max number of training iterations")
    parser.add_argument('--resume',         action='store_true', default=False, help="resume training from specified checkpoint")
    parser.add_argument('--sync_wandb',     action='store_true', default=False, help="synchronize training log with wandb")
    parser.add_argument('--export_onnx',    action='store_true', default=False, help="export policy as onnx (besides jit)")
    parser.add_argument('--debug',          action='store_true', default=False, help="enable debug mode")
    parser.add_argument('--load_run',       type=str, default=None, help="run to load, default: last run")
    parser.add_argument('--ckpt',           type=int, default=-1, help="checkpoint to load, -1 means latest")
    parser.add_argument('--use_joystick',   action='store_true', default=False, help="use joystick to provide commands")
    parser.add_argument('--joystick_type',  type=str, default='xbox', help="type of joystick: xbox, switch")
    parser.add_argument('--follow_robot',   action='store_true', default=False, help="whether the camera follows the robot during play")
    parser.add_argument('--viewer',         type=str, default='native', choices=['native', 'viser'], 
                        help="viewer backend: 'native' for simulator viewer, 'viser' for web-based 3D viewer")
    parser.add_argument('--viser_port',     type=int, default=8080, help="port for viser web server (only used with --viewer viser)")
    parser.add_argument('--motion_file',    type=str, 
                        default=None, 
                        help="motion file to load, under resources/reference_motion")
    parser.add_argument('--motion_out_dir',  type=str, default=None, help="directory to save the motion generated by the process_reference_motion script")
    parser.add_argument('--num_student',     type=int, default=None, help="[Only used in ts_depth] number of student envs to train in parallel for distillation")
    
    return parser.parse_args()

class PolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
    
    def forward(self, obs):
        return self.actor(obs)
    
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path_pt = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path_pt)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["nn_input"]
            output_names = ["nn_output"]
            dummy_input = torch.randn(1, env_cfg.env.num_observations)
            torch.onnx.export(self, dummy_input, path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterTS(torch.nn.Module):
    """Policy exporter for teacher student policies

    Attention: This module is consistent with ActorCriticTS in rsl_rl/modules/actor_critic_ts.py
               When ActorCriticTS is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.encoder = copy.deepcopy(actor_critic.history_encoder)
    
    def forward(self, obs, history):
        latent = self.encoder(history)
        x = torch.cat([obs, latent], dim=-1)
        return self.actor(x)
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["obs_input", "obs_history_input"]
            output_names = ["nn_output"]
            dummy_obs = torch.randn(1, env_cfg.env.num_observations)
            dummy_history = torch.randn(1, env_cfg.env.num_history_obs)
            torch.onnx.export(self, (dummy_obs, dummy_history), path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterEE(torch.nn.Module):
    """Policy exporter for explicit estimator policies

    Attention: This module is consistent with ActorCriticEE in rsl_rl/modules/actor_critic_ee.py
               When ActorCriticEE is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator)
    
    def forward(self, obs_history):
        estimated_state = self.estimator(obs_history)
        x = torch.cat([obs_history, estimated_state], dim=-1)
        return self.actor(x)
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        pt_path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(pt_path)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            onnx_path = os.path.join(path, filename)
            input_names = ["nn_input"]
            output_names = ["nn_output"]
            dummy_input = torch.randn(1, env_cfg.env.num_estimator_features)
            torch.onnx.export(self, dummy_input, onnx_path, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterWaQ(torch.nn.Module):
    """Policy exporter for DreamWaQ policies
    
    Attention: This module is consistent with ActorCriticDreamWaQ in rsl_rl/modules/actor_critic_dreamwaq.py
               When ActorCriticDreamWaQ is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.vae = copy.deepcopy(actor_critic.vae)
    
    def forward(self, obs, obs_history):
        vae_out = self.vae.inference(obs_history)
        x = torch.cat([obs, vae_out], dim=-1)
        return self.actor(x)
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["obs_input", "obs_history_input"]
            output_names = ["nn_output"]
            dummy_obs = torch.randn(1, env_cfg.env.num_observations)
            dummy_history = torch.randn(1, env_cfg.env.num_history_obs)
            torch.onnx.export(self, (dummy_obs, dummy_history), path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterDepth(torch.nn.Module):
    """Policy exporter for depth-based policies
    
    Attention: This module is consistent with ActorCriticDepth in rsl_rl/modules/actor_critic_depth.py
               When ActorCriticDepth is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic: ActorCriticTSDepth, train_cfg):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.cnn = copy.deepcopy(actor_critic.depth_history_encoder.cnn)
        self.combination_mlp = copy.deepcopy(actor_critic.depth_history_encoder.combination_mlp)
        self.rnn = copy.deepcopy(actor_critic.depth_history_encoder.rnn)
        self.latent_output_mlp = copy.deepcopy(actor_critic.depth_history_encoder.latent_output_mlp)
        # self.register_buffer("hidden_states", 
        #                       torch.zeros(train_cfg.policy.rnn_num_layers, 1, train_cfg.policy.rnn_hidden_size))
    
    def forward(self, obs, depth_obs, hidden_states):
        cnn_encoding = self.cnn(depth_obs)
        combined_input = torch.cat((obs.unsqueeze(0), cnn_encoding.unsqueeze(0)), dim=-1)
        combined_encoding = self.combination_mlp(combined_input)
        rnn_out, hidden_states = self.rnn(combined_encoding, hidden_states)
        latent_output = self.latent_output_mlp(rnn_out)
        x = torch.cat([obs, latent_output.squeeze(0)], dim=-1)
        actions = self.actor(x)
        return actions, hidden_states
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["obs_input", "depth_obs_input"]
            output_names = ["nn_output"]
            dummy_obs = torch.randn(1, env_cfg.env.num_observations)
            depth_cfg = env_cfg.sensor.depth_camera_config
            dummy_depth_obs = torch.randn(1, depth_cfg.resolution[0]*depth_cfg.resolution[1])
            torch.onnx.export(self, (dummy_obs, dummy_depth_obs), path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()

    def forward(self, x, hidden_state, cell_state):
        out, (h, c) = self.memory(x.unsqueeze(0), (hidden_state, cell_state))
        return self.actor(out.squeeze(0)), h, c
    
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)