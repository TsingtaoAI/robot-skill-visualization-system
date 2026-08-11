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

from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO, DeviceType
from rsl_rl.modules import ActorCritic
from rsl_rl.modules.amp_discriminator import AMPDiscriminator
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer import ReplayBuffer
from rsl_rl.utils.utils import Normalizer


# Type alias for symmetry config
SymmetryConfig = Dict[str, Any]


class PPO_AMP(PPO):
    """PPO with Adversarial Motion Priors (AMP) for learning from motion clips.
    
    Refer to: https://arxiv.org/abs/2104.02180
    """
    
    actor_critic: ActorCritic
    storage: Optional[RolloutStorage]
    
    # AMP-specific components
    discriminator: AMPDiscriminator
    amp_storage: ReplayBuffer
    amp_data: ReplayBuffer
    amp_normalizer: Optional[Normalizer]
    amp_transition: RolloutStorage.Transition
    disc_optimizer: optim.Optimizer
    symmetry_cfg: Optional[SymmetryConfig]

    def __init__(
        self,
        actor_critic: ActorCritic,
        discriminator: AMPDiscriminator,
        amp_data: ReplayBuffer,
        amp_normalizer: Optional[Normalizer],
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
        device: DeviceType = 'cpu',
        use_spo: bool = False,
        amp_replay_buffer_size: int = 100000,
        disc_lr: float = 1e-4,
        symmetry_cfg: Optional[SymmetryConfig] = None,
    ) -> None:

        super().__init__(
            actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            use_spo,
            device      
        )

        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device
        )
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # Optimizer for policy and discriminator.
        params = [{'params': self.actor_critic.parameters(), 'name': 'actor_critic'}]
        self.optimizer = optim.Adam(params, lr=learning_rate)
        disc_params = [
            {'params': self.discriminator.trunk.parameters(), 'weight_decay': 10e-4, 'name': 'amp_trunk'},
            {'params': self.discriminator.amp_linear.parameters(), 'weight_decay': 10e-2, 'name': 'amp_head'}
        ]
        self.disc_optimizer = optim.Adam(disc_params, lr=disc_lr)
        
        # symmetry config
        self.symmetry_cfg = symmetry_cfg

    def act(  # type: ignore[override]
        self, 
        obs: torch.Tensor, 
        critic_obs: torch.Tensor, 
        amp_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute actions with AMP observation recording.
        
        Args:
            obs: Actor observations. Shape: [num_envs, obs_dim]
            critic_obs: Critic observations. Shape: [num_envs, critic_obs_dim]
            amp_obs: AMP observations for discriminator. Shape: [num_envs, amp_obs_dim]
            
        Returns:
            actions: Sampled actions. Shape: [num_envs, action_dim]
        """
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions
    
    def process_env_step(  # type: ignore[override]
        self, 
        rewards: torch.Tensor, 
        dones: torch.Tensor, 
        infos: Dict[str, Any],
        amp_obs: torch.Tensor
    ) -> None:
        """Process environment step with AMP observation storage.
        
        Args:
            rewards: Rewards from environment. Shape: [num_envs]
            dones: Done flags. Shape: [num_envs]
            infos: Info dict, may contain 'time_outs'
            amp_obs: Next AMP observations. Shape: [num_envs, amp_obs_dim]
        """
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
            )

        not_done_idxs = (dones == False).nonzero().squeeze()
        self.amp_storage.insert(self.amp_transition.observations, amp_obs)

        # Record the transition
        assert self.storage is not None  # storage is initialized in init_storage()
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)
    
    def update(self) -> Tuple[float, float, float, float, float, float, Optional[float]]:  # type: ignore[override]
        """Update policy and discriminator.
        
        Returns:
            mean_value_loss: Average value function loss
            mean_surrogate_loss: Average surrogate loss
            mean_amp_loss: Average AMP discriminator loss
            mean_grad_pen_loss: Average gradient penalty loss
            mean_policy_pred: Average discriminator prediction on policy samples
            mean_expert_pred: Average discriminator prediction on expert samples
            mean_symmetry_loss: Average symmetry loss (or None if not used)
        """
        assert self.storage is not None  # storage is initialized in init_storage()
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        mean_symmetry_loss: Optional[float] = 0.0 if self.symmetry_cfg else None
        generator = self._get_data_generator()
        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        )
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        )
        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):

                obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
                    old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch = sample
                
                num_aug = 1
                original_batch_size = obs_batch.shape[0]
                
                # Perform symmetric augmentation
                if self.symmetry_cfg and self.symmetry_cfg["use_data_augmentation"]:
                    data_augmentation_func: Callable = self.symmetry_cfg["data_augmentation_func"]
                    obs_batch, actions_batch, critic_obs_batch = data_augmentation_func(
                        obs=obs_batch,
                        actions=actions_batch,
                        critic_obs=critic_obs_batch,
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                    old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                    target_values_batch = target_values_batch.repeat(num_aug, 1)
                    advantages_batch = advantages_batch.repeat(num_aug, 1)
                    returns_batch = returns_batch.repeat(num_aug, 1)
                
                loss, surrogate_loss, value_loss = self._compute_rl_loss(
                    original_batch_size, obs_batch, critic_obs_batch, actions_batch, 
                    target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, 
                    old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch
                )
                
                # Symmetry loss
                if self.symmetry_cfg:
                    # Obtain the symmetric actions
                    # Note: If we did augmentation before then we don't need to augment again
                    if not self.symmetry_cfg["use_data_augmentation"]:
                        data_augmentation_func = self.symmetry_cfg["data_augmentation_func"]
                        obs_batch, _, _ = data_augmentation_func(obs=obs_batch, actions=None, 
                                                                    critic_obs=None)
                        # Compute number of augmentations per sample
                        num_aug = int(obs_batch.shape[0] / original_batch_size)
                    
                    # Actions predicted by the actor for symmetrically-augmented observations
                    mean_actions_batch = self.actor_critic.act_inference(obs_batch.detach().clone())
                    # Compute the symmetrically augmented actions
                    # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                    # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                    # using the mean of the distribution.
                    action_mean_orig = mean_actions_batch[:original_batch_size]
                    _, actions_mean_symm_batch, _ = data_augmentation_func(
                        obs=None, actions=action_mean_orig, critic_obs=None
                    )
                    symmetry_loss = torch.nn.MSELoss()(
                        mean_actions_batch[original_batch_size:], 
                        actions_mean_symm_batch.detach()[original_batch_size:]
                    )
                    # Add the loss to the total loss
                    if self.symmetry_cfg["use_mirror_loss"]:
                        loss += self.symmetry_cfg["mirror_loss_coeff"] * symmetry_loss
                    else:
                        symmetry_loss = symmetry_loss.detach()
                
                # Discriminator loss.
                policy_state, policy_next_state = sample_amp_policy
                expert_state, expert_next_state = sample_amp_expert
                disc_loss, amp_loss, grad_pen_loss, policy_d, expert_d = self._compute_amp_loss(
                    policy_state, policy_next_state, expert_state, expert_next_state
                )
                
                # Update discriminator.
                self.disc_optimizer.zero_grad()
                disc_loss.backward()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
                self.disc_optimizer.step()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.amp_normalizer is not None:
                    self.amp_normalizer.update(policy_state.cpu().numpy())
                    self.amp_normalizer.update(expert_state.cpu().numpy())

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_amp_loss += amp_loss.item()
                mean_grad_pen_loss += grad_pen_loss.item()
                mean_policy_pred += policy_d.mean().item()
                mean_expert_pred += expert_d.mean().item()
                # Symmetry loss
                if mean_symmetry_loss is not None:
                    mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        self.storage.clear()

        return (
            mean_value_loss, mean_surrogate_loss,
            mean_amp_loss, mean_grad_pen_loss,
            mean_policy_pred, mean_expert_pred, mean_symmetry_loss
        )
    
    def _compute_rl_loss(  # type: ignore[override]
        self,
        original_batch_size: int,
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
        self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
        actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
        value_batch = self.actor_critic.evaluate(
            critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
        )
        mu_batch = self.actor_critic.action_mean[:original_batch_size]
        sigma_batch = self.actor_critic.action_std[:original_batch_size]
        entropy_batch = self.actor_critic.entropy[:original_batch_size]

        self._adjust_learning_rate(sigma_batch, old_sigma_batch, mu_batch, old_mu_batch)

        # Surrogate loss
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate_loss = self._compute_surrogate_loss(ratio, advantages_batch)

        # Value function loss
        value_loss = self._compute_value_function_loss(value_batch, returns_batch, target_values_batch)

        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
        
        return loss, surrogate_loss, value_loss
    
    def _compute_amp_loss(
        self,
        policy_state: torch.Tensor,
        policy_next_state: torch.Tensor,
        expert_state: torch.Tensor,
        expert_next_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute AMP discriminator loss.
        
        Returns:
            disc_loss: Total discriminator loss
            amp_loss: AMP classification loss
            grad_pen_loss: Gradient penalty loss
            policy_d: Discriminator output for policy samples
            expert_d: Discriminator output for expert samples
        """
        if self.amp_normalizer is not None:
            with torch.no_grad():
                policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)
        policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
        expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
        expert_loss = torch.nn.MSELoss()(
            expert_d, torch.ones(expert_d.size(), device=self.device)
        )
        policy_loss = torch.nn.MSELoss()(
            policy_d, -1 * torch.ones(policy_d.size(), device=self.device)
        )
        amp_loss = 0.5 * (expert_loss + policy_loss)
        grad_pen_loss = self.discriminator.compute_grad_pen(expert_state, expert_next_state, lambda_=10)
        disc_loss = amp_loss + grad_pen_loss
        
        return disc_loss, amp_loss, grad_pen_loss, policy_d, expert_d
