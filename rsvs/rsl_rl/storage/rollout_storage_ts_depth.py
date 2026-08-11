import torch
import numpy as np
from .rollout_storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories

class RolloutStorageTSDepth(RolloutStorage):
    class Transition(RolloutStorage.Transition):
        def __init__(self):
            super().__init__()
            self.privileged_observations = None
            self.depth_image_features = None
            self.teacher_actions = None

    def __init__(self, num_envs, num_student, num_transitions_per_env, obs_shape, 
                 privileged_obs_shape, depth_image_features_shape,
                 critic_obs_shape, actions_shape, device='cpu'):

        super().__init__(num_envs, num_transitions_per_env, obs_shape,
                         privileged_obs_shape, actions_shape, device)

        self.critic_obs_shape = critic_obs_shape
        self.num_student = num_student

        if self.privileged_observations is None:
            raise ValueError("Privileged observations are required for RolloutStorageTS")
        self.critic_observations = torch.zeros(num_transitions_per_env, num_envs, *critic_obs_shape, device=self.device)
        self.depth_image_features = torch.zeros(num_transitions_per_env, num_student, *depth_image_features_shape, device=self.device)
        self.teacher_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.saved_hidden_states = None
        
    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.privileged_observations[self.step].copy_(transition.privileged_observations)
        self.depth_image_features[self.step].copy_(transition.depth_image_features)
        self.critic_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        if transition.teacher_actions is not None:
            self.teacher_actions[self.step].copy_(transition.teacher_actions)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1
    
    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # only support GRU now
        hid = hidden_states if isinstance(hidden_states, tuple) else (hidden_states,)
        
        # FIX: Only save hidden states for student environments (first num_student envs)
        # This ensures hidden states align with depth_image_features which only stores student data
        hid_student = [h[:, :self.num_student, :] for h in hid]
        
        # initialize if needed - shape should match student envs only
        if self.saved_hidden_states is None:
            # hid_student shape: [num_layers, num_student, hidden_dim]
            # saved_hidden_states shape should be: [num_transitions, num_layers, num_student, hidden_dim]
            self.saved_hidden_states = [torch.zeros(self.observations.shape[0], h.shape[0], self.num_student, h.shape[-1], device=self.device) for h in hid_student]
        # copy the states
        for i in range(len(hid_student)):
            self.saved_hidden_states[i][self.step].copy_(hid_student[i])

    def teacher_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # use non-recurrent mini-batch generator for teacher update
        mini_batch_size = self.num_envs // num_mini_batches
        student_mini_batch_size = self.num_student // num_mini_batches
        
        student_obs = self.observations[:, 0:self.num_student]
        student_privileged_obs = self.privileged_observations[:, 0:self.num_student]
        padded_student_obs, trajectory_masks = split_and_pad_trajectories(
            student_obs, self.dones[:, 0:self.num_student])
        padded_student_privileged_obs, _ = split_and_pad_trajectories(
            student_privileged_obs, self.dones[:, 0:self.num_student])
        padded_depth_features, _ = split_and_pad_trajectories(
            self.depth_image_features, self.dones[:, 0:self.num_student])

        for epoch in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):

                # Generate batch indices
                start = i*mini_batch_size
                end = (i+1)*mini_batch_size
                start_student = i*student_mini_batch_size
                end_student = (i+1)*student_mini_batch_size
                
                obs_batch = self.observations[:, start:end]
                privileged_obs_batch = self.privileged_observations[:, start:end]
                critic_obs_batch = self.critic_observations[:, start:end]
                actions_batch = self.actions[:, start:end]
                values_batch = self.values[:, start:end]
                returns_batch = self.returns[:, start:end]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:end]
                advantages_batch = self.advantages[:, start:end]
                old_mu_batch = self.mu[:, start:end]
                old_sigma_batch = self.sigma[:, start:end]
                
                dones = self.dones[:, 0:self.num_student].squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start_student:end_student])
                last_traj = first_traj + trajectories_batch_size
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                student_obs_batch = padded_student_obs[:, first_traj:last_traj, :]
                depth_batch = padded_depth_features[:, first_traj:last_traj, :]
                student_privileged_obs_batch = padded_student_privileged_obs[:, first_traj:last_traj, :]
                
                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states ]
                # remove the tuple for GRU
                hid_batch = hid_batch[0] if len(hid_batch)==1 else hid_batch

                yield obs_batch, privileged_obs_batch, critic_obs_batch, actions_batch,\
                    values_batch, returns_batch, old_actions_log_prob_batch, advantages_batch, old_mu_batch, old_sigma_batch,\
                    student_obs_batch, student_privileged_obs_batch, depth_batch, hid_batch, masks_batch
                
                first_traj = last_traj
                
                
    def student_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        # use recurrent mini-batch generator for ts depth
        mini_batch_size = self.num_student // num_mini_batches

        # Only use the first num_student environments for the student (recurrent) update
        observations = self.observations[:, 0:self.num_student]
        privileged_observations = self.privileged_observations[:, 0:self.num_student]
        critic_observations = self.critic_observations[:, 0:self.num_student]
        actions = self.actions[:, 0:self.num_student]
        values = self.values[:, 0:self.num_student]
        returns = self.returns[:, 0:self.num_student]
        old_actions_log_prob = self.actions_log_prob[:, 0:self.num_student]
        advantages = self.advantages[:, 0:self.num_student]
        old_mu = self.mu[:, 0:self.num_student]
        old_sigma = self.sigma[:, 0:self.num_student]
        padded_obs, trajectory_masks = split_and_pad_trajectories(
            observations, self.dones[:, 0:self.num_student])
        padded_privileged_obs, _ = split_and_pad_trajectories(
            privileged_observations, self.dones[:, 0:self.num_student])
        padded_depth_features, _ = split_and_pad_trajectories(
            self.depth_image_features, self.dones[:, 0:self.num_student])

        for epoch in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):

                start_student = i*mini_batch_size
                end_student = (i+1)*mini_batch_size
                
                dones = self.dones[:, 0:self.num_student].squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start_student:end_student])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs[:, first_traj:last_traj, :]
                depth_batch = padded_depth_features[:, first_traj:last_traj, :]

                # For student update we keep [time, env, ...] shape to match
                # the recurrent policy/distribution, similar to
                # RolloutStorage.reccurent_mini_batch_generator
                privileged_obs_batch = padded_privileged_obs[:, first_traj:last_traj, :]
                critic_obs_batch = critic_observations[:, start_student:end_student]
                actions_batch = actions[:, start_student:end_student]
                values_batch = values[:, start_student:end_student]
                returns_batch = returns[:, start_student:end_student]
                old_actions_log_prob_batch = old_actions_log_prob[:, start_student:end_student]
                advantages_batch = advantages[:, start_student:end_student]
                old_mu_batch = old_mu[:, start_student:end_student]
                old_sigma_batch = old_sigma[:, start_student:end_student]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states ]
                # remove the tuple for GRU
                hid_batch = hid_batch[0] if len(hid_batch)==1 else hid_batch
                
                teacher_actions_batch = self.teacher_actions[:, 0:self.num_student][:, start_student:end_student]

                yield obs_batch, privileged_obs_batch, depth_batch, critic_obs_batch, \
                    actions_batch, values_batch, returns_batch, old_actions_log_prob_batch, \
                    advantages_batch, old_mu_batch, old_sigma_batch, hid_batch, masks_batch, teacher_actions_batch

                first_traj = last_traj