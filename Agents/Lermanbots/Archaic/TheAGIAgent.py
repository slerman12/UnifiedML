# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch

import Utils

from Blocks.Augmentations import IntensityAug
from Blocks.Encoders import CNNEncoder, LayerNormMLPEncoder
from Blocks.Actors import CategoricalCriticActor
from Blocks.Architectures.Lermanblocks.AGIGradient import AGIGradient

from Losses import QLearning


class DQNDPGAgent(torch.nn.Module):
    """Deep Q-Network, Deep Policy Gradient"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 device, log_tensorboard,  # On-boarding
                 **kwargs):
        super().__init__()

        self.device = device
        self.log_tensorboard = log_tensorboard
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps

        # Models
        self.encoder_feature_map = CNNEncoder(obs_shape, optim_lr=lr).to(device)

        self.encoder_trunk = LayerNormMLPEncoder()  # todo i like this separation of concerns

        self.critic = AGIGradient()

        # self.critic = EnsembleQCritic(self.encoder.repr_dim, feature_dim, hidden_dim, action_shape[-1],
        #                               target_tau=target_tau, optim_lr=lr, discrete=discrete).to(device)

        self.actor = CategoricalCriticActor(self.critic, stddev_schedule)

        # Data augmentation
        self.aug = IntensityAug(0.05)

        # Birth

    # "Play"
    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

            # "See"
            obs = self.encoder(obs)
            dist = self.actor(obs, self.step)

            action = dist.sample() if self.training \
                else dist.best if self.discrete \
                else dist.mean

            if self.training:
                self.step += 1

                # Explore phase
                if self.step < self.explore_steps and self.training:
                    action = torch.randint(self.actor.action_dim, size=action.shape) if self.discrete \
                        else action.uniform_(-1, 1)

            return action

    # "Dream"
    def update(self, replay):
        logs = {'episode': self.episode, 'step': self.step} if self.log_tensorboard \
            else None  # todo maybe use 'log' bool arg instead of log_tensorboard

        # "Recollect"

        batch = replay.sample()  # Can also write 'batch = next(replay)'
        obs, action, reward, discount, next_obs, *traj = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r = traj

        # "Imagine" / "Envision"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # Encode
        obs = self.encoder_feature_map(obs)
        obs = self.encoder_trunk(obs)
        with torch.no_grad():
            next_obs = self.encoder_feature_map(next_obs)
            next_obs = self.encoder_trunk(next_obs)

        if self.log_tensorboard:
            logs['batch_reward'] = reward.mean().item()

        # "Predict" / "Discern" / "Learn" / "Grow"

        # Encoder loss
        critic_loss = QLearning.ensembleQLearning(self.actor, self.critic,
                                                  obs, action, reward, discount, next_obs,
                                                  self.step, logs=logs)

        # Update encoder
        Utils.optimize(critic_loss,
                       self.encoder_feature_map,
                       self.encoder_trunk)

        return logs
