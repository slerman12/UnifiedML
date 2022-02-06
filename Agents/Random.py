# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch

import Utils

from Blocks.Encoders import CNNEncoder
from Blocks.Actors import TruncatedGaussianActor


class RandomAgent(torch.nn.Module):
    """Random Agent"""
    def __init__(self,
                 obs_shape, action_shape, trunk_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, device, log,  # On-boarding
                 ):
        super().__init__()

        self.device = device
        self.birthday = time.time()
        self.step = self.episode = 0

        self.encoder = CNNEncoder(obs_shape, optim_lr=lr)

        self.actor = TruncatedGaussianActor(self.encoder.feature_shape, trunk_dim, hidden_dim, action_shape[-1],
                                            discrete=discrete, stddev_schedule=stddev_schedule, stddev_clip=stddev_clip)

        # Birth

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device)

            # "See"
            obs = self.encoder(obs)

            Pi = self.actor(obs, self.step)

            action = Pi.sample() if self.training \
                else Pi.mean

            self.step += 1

            # Randomize
            action = action.uniform_(-1, 1)

            if self.discrete:
                action = torch.argmax(action, -1)  # Since using vector representations

            return action

    # "Dream"
    def learn(self, replay):
        return
