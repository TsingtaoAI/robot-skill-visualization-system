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

from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.base_algorithm import BaseAlgorithm
from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


# Type aliases for clarity
DeviceType = Union[str, torch.device]
ActorCriticType = ActorCritic  # Can be extended for variants like ActorCriticTS
StorageType = RolloutStorage  # Can be extended for variants like RolloutStorageTS


class PPO(BaseAlgorithm):
    """Proximal Policy Optimization algorithm.
    
    Supports both standard PPO and SPO (Simple Policy Optimization) modes.
    """
    
    # Core components
    actor_critic: ActorCriticType
    storage: Optional[StorageType]
    optimizer: optim.Optimizer
    transition: RolloutStorage.Transition
    
    # Hyperparameters
    clip_param: float
    num_learning_epochs: int
    num_mini_batches: int
    value_loss_coef: float
    entropy_coef: float
    gamma: float
    lam: float
    max_grad_norm: float
    use_clipped_value_loss: bool
    
    # Learning rate scheduling
    desired_kl: Optional[float]
    schedule: str
    learning_rate: float
    use_spo: bool
    
    # Device
    device: DeviceType
    
    def __init__(
        self,
        actor_critic: ActorCriticType,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: Optional[float] = 0.01,
        use_spo: bool = False,
        device: DeviceType = 'cpu',
    ) -> None:

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.use_spo = use_spo  # SPO (Simple Policy Optimization), refer to https://arxiv.org/abs/2401.16025

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: Tuple[int, ...],
        critic_obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
    ) -> None:
        """Initialize the rollout storage.
        
        Args:
            num_envs: Number of parallel environments
            num_transitions_per_env: Number of steps to store per environment
            actor_obs_shape: Shape of actor observations
            critic_obs_shape: Shape of critic observations
            action_shape: Shape of actions
        """
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, 
            critic_obs_shape, action_shape, self.device
        )

    def test_mode(self) -> None:
        self.actor_critic.test()
    
    def train_mode(self) -> None:
        self.actor_critic.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Compute actions for given observations.
        
        Args:
            obs: Actor observations. Shape: [num_envs, obs_dim]
            critic_obs: Critic observations. Shape: [num_envs, critic_obs_dim]
            
        Returns:
            actions: Sampled actions. Shape: [num_envs, action_dim]
        """
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(
        self, 
        rewards: torch.Tensor, 
        dones: torch.Tensor, 
        infos: Dict[str, Any]
    ) -> None:
        """Process environment step results.
        
        Args:
            rewards: Rewards from environment. Shape: [num_envs]
            dones: Done flags. Shape: [num_envs]
            infos: Info dict, may contain 'time_outs' for bootstrapping
        """
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        assert self.storage is not None  # storage is initialized in init_storage()
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        """Compute returns and advantages using GAE.
        
        Args:
            last_critic_obs: Final critic observations for bootstrapping. Shape: [num_envs, critic_obs_dim]
        """
        # 'last' here means 'final' in the rollout storage
        assert self.storage is not None  # storage is initialized in init_storage()
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self) -> Tuple[float, float]:
        """Update policy using collected experiences.
        
        Returns:
            mean_value_loss: Average value function loss
            mean_surrogate_loss: Average surrogate loss
        """
        assert self.storage is not None  # storage is initialized in init_storage()
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        generator = self._get_data_generator()
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:

            loss, surrogate_loss, value_loss = self._compute_rl_loss(
                obs_batch, critic_obs_batch, actions_batch, target_values_batch, 
                advantages_batch, returns_batch, old_actions_log_prob_batch, 
                old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
    
    def _get_data_generator(self) -> Iterator[Tuple]:
        """Get the appropriate data generator based on network type."""
        assert self.storage is not None  # storage is initialized in init_storage()
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        return generator
    
    def _compute_rl_loss(
        self,
        obs_batch: torch.Tensor,
        critic_obs_batch: torch.Tensor,
        actions_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
        advantages_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        old_actions_log_prob_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
        hid_states_batch: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        masks_batch: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the RL loss for a single mini-batch.
        
        Returns:
            Tuple of (total_loss, surrogate_loss, value_loss)
        """
        self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        value_batch = self.actor_critic.evaluate(
            critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
        )
        mu_batch = self.actor_critic.action_mean
        sigma_batch = self.actor_critic.action_std
        entropy_batch = self.actor_critic.entropy

        self._adjust_learning_rate(sigma_batch, old_sigma_batch, mu_batch, old_mu_batch)

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate_loss = self._compute_surrogate_loss(ratio, advantages_batch)

        # Value function loss
        value_loss = self._compute_value_function_loss(value_batch, returns_batch, target_values_batch)

        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
        
        return loss, surrogate_loss, value_loss
    
    def _compute_surrogate_loss(
        self, 
        ratio: torch.Tensor, 
        advantages: torch.Tensor
    ) -> torch.Tensor:
        """Compute the PPO or SPO surrogate loss."""
        if self.use_spo:  # simple policy optimization
            surrogate_loss = -(
                torch.squeeze(advantages) * ratio -
                torch.abs(torch.squeeze(advantages)) * torch.pow(ratio - 1.0, 2) / (2.0 * self.clip_param)
            ).mean()
        else:  # proximal policy optimization
            surrogate = -torch.squeeze(advantages) * ratio
            surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        return surrogate_loss
    
    def _adjust_learning_rate(
        self,
        sigma_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
        mu_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
    ) -> None:
        """Adjust learning rate based on KL divergence for adaptive scheduling."""
        if self.desired_kl is not None and self.schedule == 'adaptive':
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.e-5) + 
                    (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / 
                    (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1
                )
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
    
    def _compute_value_function_loss(
        self,
        value_batch: torch.Tensor,
        returns_batch: torch.Tensor,
        target_values_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the value function loss."""
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()
        return value_loss