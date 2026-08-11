import time
import os
from collections import deque
import statistics

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from rsl_rl.algorithms import PPO_TSDepth
from rsl_rl.modules import ActorCriticTSDepth
from rsl_rl.env import VecEnv
from .on_policy_runner import OnPolicyRunner
from rsl_rl.storage import RolloutStorageTSDepth

    
class TSDepthRunner(OnPolicyRunner):

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        """
        Runner for Teacher-Student learning with depth image observation.
        Args:
            env (VecEnv): vectorized environment.
            train_cfg (_type_): training configuration dictionary
            log_dir (_type_, optional): log directory. Defaults to None.
            device (str, optional): device to use. Defaults to 'cpu'.
        """
        super().__init__(env, train_cfg, log_dir, device)
    
    def _init_agent_and_algo(self):
        assert self.cfg["policy_class_name"] == "ActorCriticTSDepth"
        actor_critic_class = eval(self.cfg["policy_class_name"])

        actor_critic : ActorCriticTSDepth = actor_critic_class(
                                                    self.env.num_obs,
                                                    self.env.num_actions,
                                                    self.env.num_privileged_obs,
                                                    self.env.num_latent_dims,
                                                    self.env.num_critic_obs,
                                                    self.env.depth_image_resolution,
                                                    **self.policy_cfg).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"])
        self.alg: PPO_TSDepth = alg_class(actor_critic, device=self.device,
                                          **self.alg_cfg,
                                          num_student=self.env.num_student)
    
    def _init_storage(self):
        self.storage_full = RolloutStorageTSDepth(
            self.env.num_envs, self.env.num_student, self.num_steps_per_env,
            [self.env.num_obs], [self.env.num_privileged_obs],
            self.env.depth_image_features_shape, [self.env.num_critic_obs],
            [self.env.num_actions], self.device)
        self.alg.init_storage(self.storage_full)
        self.current_num_envs = self.env.num_envs
        
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._pre_learn(init_at_random_ep_len)
        obs, privileged_obs, depth_image_features, critic_obs = self.env.get_observations()
        obs, privileged_obs, depth_image_features, critic_obs = obs.to(self.device), privileged_obs.to(self.device), \
            depth_image_features.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.current_num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.current_num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, privileged_obs, depth_image_features, critic_obs)
                    obs, privileged_obs, depth_image_features, critic_obs, rewards, dones, infos = self.env.step(actions)
                    obs, privileged_obs, depth_image_features, rewards, dones, critic_obs = obs.to(self.device), \
                        privileged_obs.to(self.device), depth_image_features.to(self.device), rewards.to(self.device), dones.to(self.device), critic_obs.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.log_dir is not None:
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

                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_latent_reconstruction_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            # if it % 5 == 0:
            #     self.alg.actor_critic.detach_hidden_states()

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.current_num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.current_num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/latent_reconstruction', locs['mean_latent_reconstruction_loss'], locs['it'])
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

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Latent reconstruction loss:':>{pad}} {locs['mean_latent_reconstruction_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Latent reconstruction loss:':>{pad}} {locs['mean_latent_reconstruction_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)