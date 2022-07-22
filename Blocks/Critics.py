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
    """
    MLP-based Critic network, employs ensemble Q learning,
    returns a Normal distribution over the ensemble.
    """
    def __init__(self, repr_shape, trunk_dim, hidden_dim, action_spec, trunk=None, Q_head=None, ensemble_size=2,
                 discrete=False, ignore_obs=False, optim=None, scheduler=None, lr=None, lr_decay_epochs=None,
                 weight_decay=None, ema_decay=None):
        super().__init__()

        self.discrete = discrete
        self.num_actions = action_spec.num_actions or 1  # n, or undefined n'
        self.action_dim = math.prod(action_spec.shape)  # d
        self.ignore_obs = ignore_obs

        self.low, self.high = action_spec.low, action_spec.high

        assert not (ignore_obs and discrete), "Discrete actor always requires observation, cannot ignore_obs"

        in_dim = math.prod(repr_shape)
        out_dim = self.num_actions * self.action_dim if discrete else 1

        self.trunk = Utils.instantiate(trunk, input_shape=repr_shape, output_dim=trunk_dim) or nn.Sequential(
            nn.Flatten(), nn.Linear(in_dim, trunk_dim), nn.LayerNorm(trunk_dim), nn.Tanh())  # Not used if ignore_obs

        in_shape = action_spec.shape if ignore_obs else [trunk_dim + (0 if discrete
                                                                      else self.num_actions * self.action_dim)]

        self.Q_head = Utils.Ensemble([Utils.instantiate(Q_head, i, input_shape=in_shape, output_dim=out_dim) or
                                      MLP(in_shape, out_dim, hidden_dim, 2) for i in range(ensemble_size)], 0)  # e

        # Discrete actions are known a priori
        if discrete:
            action = torch.cartesian_prod(*[torch.arange(self.num_actions)] * self.action_dim)  # [n^d, d]

            if self.low or self.high:
                action = action / self.num_actions * (self.high - self.low) + self.low  # Normalize

            self.register_buffer('action', action.view(-1, self.action_dim))

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).eval()

    def forward(self, obs, action=None, context=None):
        batch_size = obs.shape[0]

        h = torch.empty((batch_size, 0), device=action.device) if self.ignore_obs \
            else self.trunk(obs)

        if context is None:
            context = torch.empty(0, device=h.device)

        # Ensemble

        if self.discrete:
            # All actions' Q-values
            Qs = self.Q_head(h, context).unflatten(-1, [self.num_actions, self.action_dim])  # [e, b, n, d]

            # TODO Either sample num actions Qs/actions, or use all num_actions Qs/samples: max num_actions
            #  Too expensive! Leave nxd; row-wise sample/argmax Creator; index into n x d
            Qs = Utils.batched_cartesian_prod(Qs.unbind(-1)).mean(-1).flatten(2)  # [e, b, n^d]

            if action is None:
                action = self.action.expand(*Qs[0].shape, self.action_dim)  # [b, n^d, d]
            else:
                # Un-normalize to indices
                if self.low and self.high:
                    action = (action - self.low) / (self.high - self.low) * self.num_actions

                # Q values for a discrete action
                Qs = Utils.gather_indices(Qs, action)  # [e, b, 1]
        else:
            assert action is not None and action.shape[-1] == self.num_actions * self.action_dim, \
                f'action with dim={self.num_actions * self.action_dim} needed for continuous space'

            action = action.reshape(batch_size, -1, self.num_actions * self.action_dim)  # [b, n', n * d]

            h = h.unsqueeze(1).expand(*action.shape[:-1], -1)

            # Q-values for continuous action(s)
            Qs = self.Q_head(h, action, context).squeeze(-1)  # [e, b, n']

        # Dist
        stddev, mean = torch.std_mean(Qs, dim=0)
        Q = Normal(mean, stddev.nan_to_num() + 1e-8)
        Q.__dict__.update({'Qs': Qs,
                           'action': action})

        return Q
