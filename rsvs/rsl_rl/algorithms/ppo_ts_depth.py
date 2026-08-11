import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.storage import RolloutStorageTSDepth
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.utils import unpad_trajectories
from rsl_rl.modules import ActorCriticTSDepth

'''
PPO with teacher-student architecture, for policy with depth image observation
'''


class PPO_TSDepth(PPO):
    actor_critic: ActorCriticTSDepth
    
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 use_spo=False,
                 device='cpu',
                 num_student=1,
                 encoder_lr=1.e-4
                 ):

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
            device,
        )

        self.num_student = num_student
        self.encoder_lr = encoder_lr

        self.actor_critic: ActorCriticTSDepth = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.teacher_params = list(self.actor_critic.actor.parameters()) + \
                            list(self.actor_critic.critic.parameters()) + \
                            [self.actor_critic.std]
        self.teacher_optimizer = optim.Adam([
            {'params': self.teacher_params, 'lr': learning_rate},
            {'params': list(self.actor_critic.privilege_encoder.parameters()), 
             'lr': self.encoder_lr * 0.5}
        ])
        self.student_params = list(self.actor_critic.depth_history_encoder.parameters())
        self.student_optimizer = optim.Adam(
            self.student_params, lr=self.encoder_lr)
        self.transition = RolloutStorageTSDepth.Transition()

    def init_storage(self, storage):
        self.storage: RolloutStorageTSDepth = storage

    def act(self, obs, privileged_obs, depth_image_features, critic_obs):          
        self.transition.hidden_states = self.actor_critic.get_hidden_states()
        _ = self.actor_critic.depth_history_encoder(obs[0:self.num_student], depth_image_features)
        self.transition.actions = self.actor_critic.act(obs, None, privileged_obs,
                                                        "teacher", None, None).detach()
        self.transition.teacher_actions = None
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.observations = obs.detach()
        self.transition.privileged_observations = privileged_obs.detach()
        self.transition.depth_image_features = depth_image_features.detach()
        self.transition.critic_observations = critic_obs.detach()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones[:self.num_student])

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_latent_reconstruction_loss = 0
        # FIX: Separate teacher and student updates completely
        # First, update teacher for all mini-batches
        generator = self.storage.teacher_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, privileged_obs_batch, critic_obs_batch, actions_batch, \
            target_values_batch, returns_batch, old_actions_log_prob_batch, \
            advantages_batch, old_mu_batch, old_sigma_batch,\
            student_obs_batch, student_privileged_obs_batch, depth_features_batch,\
            hid_states_batch, masks_batch in generator:
            
            # Teacher update only
            self.actor_critic.act(
                obs_batch, None, privileged_obs_batch, "teacher", None, None)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            entropy_batch = self.actor_critic.entropy
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            
            ## Teacher KL, adapt learning rate
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) +
                        (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) /
                        (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(
                            1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(
                            1e-2, self.learning_rate * 1.5)

                    self.teacher_optimizer.param_groups[0]['lr'] = self.learning_rate

            ## Surrogate loss
            ratio = torch.exp(actions_log_prob_batch -
                            torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                            1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(
                    value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = self.value_loss_coef * value_loss + surrogate_loss \
                    - self.entropy_coef * entropy_batch.mean()
            # Gradient step
            self.teacher_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.teacher_params, self.max_grad_norm)
            self.teacher_optimizer.step()
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
        
        # Then, update student encoder separately with frozen teacher
        # This ensures stable targets for student training
        generator = self.storage.teacher_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, privileged_obs_batch, critic_obs_batch, actions_batch, \
            target_values_batch, returns_batch, old_actions_log_prob_batch, \
            advantages_batch, old_mu_batch, old_sigma_batch,\
            student_obs_batch, student_privileged_obs_batch, depth_features_batch,\
            hid_states_batch, masks_batch in generator:
            
            # Student encoder update only (teacher is frozen)
            latent = self.actor_critic.depth_history_encoder(student_obs_batch, 
                                                            depth_features_batch, 
                                                            hidden_states=hid_states_batch,
                                                            masks=masks_batch)
                
            with torch.no_grad():
                # Target from frozen privilege encoder (no gradient)
                unpadded_student_privileged_obs = unpad_trajectories(student_privileged_obs_batch, masks_batch)
                latent_targets = self.actor_critic.privilege_encoder(unpadded_student_privileged_obs)

            latent_reconstruction_loss = nn.functional.mse_loss(
                latent, latent_targets)
            
            loss = latent_reconstruction_loss
            self.student_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.student_params, self.max_grad_norm)
            self.student_optimizer.step()
            
            mean_latent_reconstruction_loss += latent_reconstruction_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_latent_reconstruction_loss /= num_updates
        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss, mean_latent_reconstruction_loss