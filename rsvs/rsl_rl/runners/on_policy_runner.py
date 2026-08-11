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

from __future__ import annotations

import time
import os
from collections import deque
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import wandb
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv


# Type aliases for configuration dictionaries
RunnerConfig = Dict[str, Any]
AlgorithmConfig = Dict[str, Any]
PolicyConfig = Dict[str, Any]
TrainConfig = Dict[str, Any]


class OnPolicyRunner:
    """On-policy RL training runner for PPO-style algorithms."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: TrainConfig,
        log_dir: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initialize the on-policy runner.

        Args:
            env: Vectorized environment for training.
            train_cfg: Training configuration containing runner, algorithm, and policy configs.
            log_dir: Directory for logging and saving models.
            device: Device to run training on (e.g., 'cpu', 'cuda').
        """
        self.cfg: RunnerConfig = train_cfg["runner"]
        self.alg_cfg: AlgorithmConfig = train_cfg["algorithm"]
        self.policy_cfg: PolicyConfig = train_cfg["policy"]
        self.all_cfg: TrainConfig = train_cfg
        self.wandb_run_name: str = (
            self.cfg["experiment_name"]
            + "_"
            + datetime.now().strftime("%b%d_%H-%M-%S")
            + "_"
            + self.cfg["run_name"]
        )
        self.device: torch.device = torch.device(device)
        self.env: VecEnv = env
        self._init_agent_and_algo()
        self.num_steps_per_env: int = self.cfg["num_steps_per_env"]
        self.save_interval: int = self.cfg["save_interval"]
        self._init_storage()

        # Log
        self.log_dir: Optional[str] = log_dir
        self.sync_wandb: bool = self.cfg.get("sync_wandb", False)
        self.writer: Optional[SummaryWriter] = None
        self.tot_timesteps: int = 0
        self.tot_time: float = 0.0
        self.current_learning_iteration: int = 0

        self.env.reset()
    
    def _init_agent_and_algo(self) -> None:
        """Initialize the actor-critic network and PPO algorithm."""
        if self.env.num_privileged_obs is not None:
            num_critic_obs: int = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCritic = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
    
    def _init_storage(self) -> None:
        """Initialize the rollout storage for the algorithm."""
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env, 
            (self.env.num_obs,),
            (self.env.num_privileged_obs,), 
            (self.env.num_actions,),
        )
    
    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run the training loop for a specified number of iterations.

        Args:
            num_learning_iterations: Number of learning iterations to run.
            init_at_random_ep_len: Whether to initialize episode lengths randomly.
        """
        self._pre_learn(init_at_random_ep_len)
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos: List[Dict[str, Any]] = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                assert self.log_dir is not None
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        assert self.log_dir is not None
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def _pre_learn(self, init_at_random_ep_len: bool) -> None:
        """Prepare for training by initializing logging and episode buffers.

        Args:
            init_at_random_ep_len: Whether to randomize initial episode lengths.
        """
        if self.log_dir is not None and self.writer is None:
            if self.sync_wandb:
                wandb.init(
                    project="LeggedGym-Ex",
                    name=self.wandb_run_name,
                    sync_tensorboard=True,
                    config=self.all_cfg,
                )
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
    
    def log(
        self,
        locs: Dict[str, Any],
        width: int = 80,
        pad: int = 35,
    ) -> None:
        """Log training metrics to tensorboard and console.

        Args:
            locs: Dictionary containing iteration metrics and buffers.
            width: Width of the log output.
            pad: Padding for log formatting.
        """
        assert self.writer is not None
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str_iter = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_iter.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str_iter.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(
        self,
        path: str,
        infos: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save the model checkpoint to disk.

        Args:
            path: File path to save the checkpoint.
            infos: Optional additional information to save with the checkpoint.
        """
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Load a model checkpoint from disk.

        Args:
            path: File path to load the checkpoint from.
            load_optimizer: Whether to load the optimizer state.

        Returns:
            Optional infos dict stored in the checkpoint.
        """
        loaded_dict = torch.load(path, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(
        self,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get the inference policy function.

        Args:
            device: Device to run inference on. If None, uses current device.

        Returns:
            Callable that takes observations and returns actions.
        """
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
