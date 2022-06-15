# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

import torch
from torch import nn
from torch.distributions import Categorical

from Distributions import TruncatedNormal

from Blocks.Architectures.MLP import MLP

import Utils


class EnsembleGaussianActor(nn.Module):
    def __init__(self, repr_shape, trunk_dim, hidden_dim, action_shape, trunk=None, pi_head=None, ensemble_size=2,
                 stddev_schedule=None, stddev_clip=None, optim=None, scheduler=None, lr=None, lr_decay_epochs=None,
                 weight_decay=None, ema_decay=None):
        super().__init__()

        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        action_dim = math.prod(action_shape)

        in_dim = math.prod(repr_shape)
        out_dim = action_dim * 2 if stddev_schedule is None else action_dim

        self.trunk = Utils.instantiate(trunk, input_shape=repr_shape, output_dim=trunk_dim) or nn.Sequential(
            nn.Linear(in_dim, trunk_dim), nn.LayerNorm(trunk_dim), nn.Tanh())

        self.Pi_head = Utils.Ensemble([Utils.instantiate(pi_head, i, input_shape=[trunk_dim], output_dim=out_dim)
                                       or MLP(trunk_dim, out_dim, hidden_dim, 2) for i in range(ensemble_size)])

        # Initializes model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        self.ema_decay = ema_decay
        self.update_ema_params()

    def update_ema_params(self):
        if not hasattr(self, 'ema') and self.ema_decay:
            self.ema = copy.deepcopy(self).eval()

        Utils.param_copy(self, self.ema, self.ema_decay)

    def forward(self, obs, step=None):
        obs = self.trunk(obs)

        if self.stddev_schedule is None or step is None:
            mean_tanh, log_stddev = self.Pi_head(obs).squeeze(1).chunk(2, dim=-1)
            stddev = torch.exp(log_stddev)
        else:
            mean_tanh = self.Pi_head(obs).squeeze(1)
            stddev = torch.full_like(mean_tanh,
                                     Utils.schedule(self.stddev_schedule, step))

        mean = torch.tanh(mean_tanh)
        Pi = TruncatedNormal(mean, stddev, low=-1, high=1, stddev_clip=self.stddev_clip)

        return Pi


class CategoricalCriticActor(nn.Module):  # a.k.a. "Creator"
    """Categorically samples over continuous or discrete actions based on critic Q-values"""
    def __init__(self, entropy_schedule=1):
        super().__init__()

        self.entropy_schedule = entropy_schedule

    def forward(self, Q, step=None, exploit_temp=1, sample_q=False, action=None, action_log_prob=0):
        # Sample q or mean
        q = Q.rsample() if sample_q else Q.mean if hasattr(Q, 'mean') else Q.best

        # Exploit Q value vs. explore Q uncertainty, e.g., UCB exploration
        # Standard deviation of Q ensemble might be a good heuristic for uncertainty for exploration
        u = exploit_temp * q + (1 - exploit_temp) * Q.stddev
        u_logits = u - u.max(dim=-1, keepdim=True)[0]

        # Entropy of action selection
        entropy_temp = Utils.schedule(self.entropy_schedule, step)

        Psi = Categorical(logits=u_logits / entropy_temp + action_log_prob)

        best_u, best_ind = torch.max(u, -1)
        best_action = Utils.gather_indices(Q.action if action is None else action, best_ind.unsqueeze(-1), 1).squeeze(1)

        sample = Psi.sample

        def action_sampler(sample_shape=torch.Size()):
            i = sample(sample_shape)
            return Utils.gather_indices(Q.action if action is None else action, i.unsqueeze(-1), 1).squeeze(1)

        Psi.__dict__.update({'best': best_action,
                             'best_u': best_u,
                             'sample_ind': sample,
                             'sample': lambda x=torch.Size(): action_sampler(x).detach(),
                             'rsample': action_sampler,
                             'Q': Q,
                             'q': q,
                             'action': Q.action if action is None else action,
                             'u': u})
        return Psi
