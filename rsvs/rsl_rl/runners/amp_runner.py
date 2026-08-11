from __future__ import annotations


import time
import os
from collections import deque
import statistics
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from .on_policy_runner import OnPolicyRunner, TrainConfig
from rsl_rl.algorithms import PPO_AMP
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv
from rsl_rl.modules.amp_discriminator import AMPDiscriminator
from legged_gym.utils.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer


class AMPRunner(OnPolicyRunner):
    """Runner for training with Adversarial Motion Priors (AMP)."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: TrainConfig,
        log_dir: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)


    def _init_agent_and_algo(self) -> None:
        """Initialize the AMP actor-critic and PPO_AMP algorithm."""
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
        amp_data = AMPLoader(
            self.device, 
            num_dof=self.env.num_actions,
            num_key_bodies=len(self.env.simulator.key_body_indices),  # type: ignore[attr-defined]
            time_between_frames=self.env.dt,  # type: ignore[attr-defined]
            preload_transitions=True,
            num_preload_transitions=self.cfg['amp_num_preload_transitions'],
            motion_files=self.cfg["amp_motion_files"]
        )
        amp_normalizer = Normalizer(amp_data.observation_dim)
        discriminator = AMPDiscriminator(
            amp_data.observation_dim * 2,
            self.cfg['amp_reward_coef'],
            self.cfg['amp_discr_hidden_dims'],
            self.device,
            self.cfg['amp_task_reward_lerp']
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO_AMP = alg_class(
            actor_critic, 
            discriminator, 
            amp_data, 
            amp_normalizer, 
            device=self.device,
            **self.alg_cfg
        )

    def learn(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Run AMP training loop for a specified number of iterations.


        Args:
            num_learning_iterations: Number of learning iterations to run.
            init_at_random_ep_len: Whether to initialize episode lengths randomly.
        """
        self._pre_learn(init_at_random_ep_len)
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        amp_obs = self.env.get_amp_observations()  # type: ignore[attr-defined]
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, amp_obs = (
            obs.to(self.device),
            critic_obs.to(self.device),
            amp_obs.to(self.device),
        )
        self.alg.actor_critic.train()
        self.alg.discriminator.train()

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
                    actions = self.alg.act(obs, critic_obs, amp_obs)
                    obs, privileged_obs, rewards, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions)  # type: ignore[misc]
                    next_amp_obs = self.env.get_amp_observations()  # type: ignore[attr-defined]

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        next_amp_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    # Account for terminal states. Use amp states before reset_idx
                    next_amp_obs_with_term = torch.clone(next_amp_obs)
                    next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    rewards, amp_reward = self.alg.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer
                    )
                    amp_obs = torch.clone(next_amp_obs)
                    self.alg.process_env_step(rewards, dones, infos, next_amp_obs_with_term)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            # add amp reward to episode info for terminal logging
                            infos['episode']['rew_amp'] = amp_reward / self.env.dt  # type: ignore[attr-defined]
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
            
            mean_value_loss, mean_surrogate_loss, \
                mean_amp_loss, mean_grad_pen_loss, \
                    mean_policy_pred, mean_expert_pred, mean_symmetry_loss = self.alg.update()
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


    def log(
        self,
        locs: Dict[str, Any],
        width: int = 80,
        pad: int = 35,
    ) -> None:
        """Log AMP training metrics to tensorboard and console.


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
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('AMP/policy_pred', locs['mean_policy_pred'], locs['it'])
        self.writer.add_scalar('AMP/expert_pred', locs['mean_expert_pred'], locs['it'])
        if locs['mean_symmetry_loss'] is not None:
            self.writer.add_scalar('Loss/symmetry_loss', locs['mean_symmetry_loss'], locs['it'])
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
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'Mean symmetry loss:':>{pad}} {locs['mean_symmetry_loss']:.4f}\n"""
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
        """Save the AMP model checkpoint to disk.

        Args:
            path: File path to save the checkpoint.
            infos: Optional additional information to save with the checkpoint.
        """
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos
        }, path)

    def load(
        self,
        path: str,
        load_optimizer: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Load an AMP model checkpoint from disk.

        Args:
            path: File path to load the checkpoint from.
            load_optimizer: Whether to load the optimizer state.

        Returns:
            Optional infos dict stored in the checkpoint.
        """
        loaded_dict = torch.load(path, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']
