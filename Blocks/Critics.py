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
    def __init__(self, repr_shape, trunk_dim, hidden_dim, action_shape, trunk=None, q_head=None, ensemble_size=2,
                 discrete=False, ignore_obs=False, optim=None, scheduler=None, lr=None, lr_decay_epochs=None,
                 weight_decay=None, ema_decay=None):
        super().__init__()

        self.discrete = discrete
        self.action_shape = action_shape
        self.action_dim = 0 if discrete else math.prod(self.action_shape)  # d

        assert not (ignore_obs and discrete), "Discrete actor always requires observation, cannot ignore_obs"
        self.ignore_obs = ignore_obs

        in_dim = math.prod(repr_shape)
        out_dim = self.action_dim if discrete else 1

        self.trunk = Utils.instantiate(trunk, input_shape=repr_shape, output_dim=trunk_dim) or nn.Sequential(
                nn.Linear(in_dim, trunk_dim), nn.LayerNorm(trunk_dim), nn.Tanh())  # Not used if ignore_obs

        in_shape = action_shape if ignore_obs else [trunk_dim + self.action_dim]

        self.Q_head = Utils.Ensemble([Utils.instantiate(q_head, i, input_shape=in_shape, output_dim=out_dim) or
                                      MLP(in_shape, out_dim, hidden_dim, 2) for i in range(ensemble_size)], 0)

        # Initializes model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        self.ema_decay = ema_decay
        self.update_ema_params()

    def update_ema_params(self):
        if not hasattr(self, 'ema') and self.ema_decay:
            self.ema = copy.deepcopy(self).eval()

        Utils.param_copy(self, self.ema, self.ema_decay)

    def forward(self, obs, action=None, context=None):
        batch_size = obs.shape[0]

        h = torch.empty((batch_size, 0), device=action.device) if self.ignore_obs \
            else self.trunk(obs).to(action.device)

        if context is None:
            context = torch.empty(0, device=h.device)

        # Ensemble

        if self.discrete:
            # All actions' Q-values
            Qs = self.Q_head(h, context)  # [e, b, n]

            if action is None:
                action = torch.arange(math.prod(self.action_shape), device=obs.device).expand_as(Qs[0])  # [b, n]
            else:
                # Q values for a discrete action
                Qs = Utils.gather_indices(Qs, action)  # [e, b, 1]
        else:
            assert action is not None and \
                   action.shape[-1] == self.action_dim, f'action with dim={self.action_dim} needed for continuous space'

            action = action.reshape(batch_size, -1, self.action_dim)  # [b, n, d]

            h = h.unsqueeze(1).expand(*action.shape[:-1], -1)

            # Q-values for continuous action(s)
            Qs = self.Q_head(h, action, context).squeeze(-1)  # [e, b, n]

        # Dist
        stddev, mean = torch.std_mean(Qs, dim=0)
        Q = Normal(mean, stddev + 1e-8)
        Q.__dict__.update({'Qs': Qs,
                           'action': action})

        return Q
