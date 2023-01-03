# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

import Utils

from Agents import DrQV2Agent

from Blocks.Critics import EnsemblePROCritic

from Losses.QLearning import ensembleQLearning


class PROAgent(DrQV2Agent):
    """Policy Ratio Optimization (PRO), A.K.A. the Proportionality Agent"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # architecture
                 target_tau, stddev_schedule, stddev_clip,  # models
                 lr, update_every_steps,  # optimization
                 num_expl_steps,  # exploration
                 discrete, device, use_tb  # on-boarding
                 ):
        super().__init__(
            obs_shape, action_shape, feature_dim, hidden_dim,  # architecture
            stddev_schedule, stddev_clip, target_tau,  # models
            lr, update_every_steps,  # optimization
            num_expl_steps,  # exploration
            discrete, device, use_tb  # on-boarding
        )

        self.critic = EnsemblePROCritic(self.encoder.repr_dim, feature_dim, hidden_dim, ensemble_size=2,
                                        target_tau=target_tau, optim_lr=lr, discrete=discrete).to(device)

    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, *traj = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r = traj

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # policy
        dist = self.actor(obs, step)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # critic loss
        critic_loss = ensembleQLearning(self.actor, self.critic,
                                        obs, action, reward, discount, next_obs, step, dist,
                                        logs=metrics if self.use_tb else None)

        # clear grads
        self.critic.optim.zero_grad(set_to_none=True)
        self.actor.optim.zero_grad(set_to_none=True)
        self.encoder.optim.zero_grad(set_to_none=True)

        # update critic, actor, encoder
        critic_loss.backward()
        self.critic.optim.step()
        self.actor.optim.step()
        self.encoder.optim.step()

        # update critic target
        self.critic.update_target()

        return metrics
