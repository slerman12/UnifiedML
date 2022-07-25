# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

import torch
from torch import nn
from torch.distributions import Normal

from Blocks.Architectures.MLP import MLP

import Utils


class EnsembleQCritic(nn.Module):
    def __init__(self, repr_shape, trunk_dim, hidden_dim, action_spec, trunk=None, Q_head=None, ensemble_size=2,
                 discrete=False, ignore_obs=False, optim=None, scheduler=None, lr=None, lr_decay_epochs=None,
                 weight_decay=None, ema_decay=None):
        super().__init__()

        self.discrete = discrete
        self.num_actions = action_spec.num_actions or 1  # n, or undefined n'
        self.action_dim = math.prod(action_spec.shape)  # d
        self.ignore_obs = ignore_obs and not discrete  # Discrete critic always requires observation

        self.low, self.high = action_spec.low, action_spec.high

        in_dim = math.prod(repr_shape)
        out_dim = self.num_actions * self.action_dim if discrete else 1

        self.trunk = Utils.instantiate(trunk, input_shape=repr_shape, output_dim=trunk_dim) or nn.Sequential(
            nn.Flatten(), nn.Linear(in_dim, trunk_dim), nn.LayerNorm(trunk_dim), nn.Tanh())  # Not used if ignore_obs

        in_shape = action_spec.shape if ignore_obs else [trunk_dim + (0 if discrete
                                                                      else self.num_actions * self.action_dim)]

        # Ensemble
        self.Q_head = Utils.Ensemble([Utils.instantiate(Q_head, i, input_shape=in_shape, output_dim=out_dim) or
                                      MLP(in_shape, out_dim, hidden_dim, 2) for i in range(ensemble_size)])  # e

        # Discrete actions are known a priori
        if discrete and action_spec.discrete:
            action = torch.cartesian_prod(*[torch.arange(self.num_actions)] * self.action_dim).view(-1, self.action_dim)

            self.register_buffer('action', self.normalize(action))  # [n^d, d]

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).eval()

    def forward(self, obs, action=None, Pi=None):
        batch_size = obs.shape[0]

        h = torch.empty((batch_size, 0), device=action.device) if self.ignore_obs \
            else self.trunk(obs)

        if self.discrete:
            assert getattr(self, 'action', action is not None or Pi), \
                'Continuous Env: action or sampling policy needed by discrete Critic.'

            # All actions' Q-values
            All_Qs = getattr(Pi, 'All_Qs',
                             self.Q_head(h).unflatten(-1, [self.num_actions, self.action_dim]))  # [b, e, n, d]

            # Given action, or all actions, or sample a subset of the action space
            action = action or getattr(self, 'action',  Pi.sample(self.num_actions)
                                       .expand(batch_size, -1, self.action_dim))  # [b, n^d, d] or [b, n', d]

            # Q values for discrete action(s)
            Qs = Utils.gather_indices(All_Qs, self.to_indices(action), -2, -2).mean(-1)  # [b, e, 1]
        else:
            assert action is not None, f'action needed by continuous action-space Critic.'

            action = action.reshape(batch_size, -1, self.num_actions * self.action_dim)  # [b, n', n * d]

            h = h.unsqueeze(1).expand(*action.shape[:-1], -1)

            # Q-values for continuous action(s)
            Qs = self.Q_head(h, action).squeeze(-1)  # [b, e, n']

        # Dist
        stddev, mean = torch.std_mean(Qs, dim=0)
        Q = Normal(mean, stddev.nan_to_num() + 1e-8)

        return Q

    def normalize(self, action):
        return action / (self.num_actions - 1) * (self.high - self.low) + self.low if self.low or self.high \
            else action  # Normalize -> [low, high]

    def to_indices(self, action):
        action = action.view(len(action), -1, self.action_dim)  # [b, n', d]

        return (action - self.low) / (self.high - self.low) * (self.num_actions - 1) if self.low or self.high \
            else action  # Inverse of normalize -> indices
