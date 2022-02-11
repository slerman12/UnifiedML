# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

from hydra.utils import call

import torch
from torch import nn
from torch.distributions import Categorical

from Distributions import TruncatedNormal

import Utils

from Blocks.Architectures.MLP import MLP


class GaussianActorEnsemble(nn.Module):
    def __init__(self, repr_shape, trunk_dim, hidden_dim, action_dim, recipe, l2_norm=False, ensemble_size=2,
                 discrete=False, stddev_schedule=None, stddev_clip=None,
                 ema_tau=None, optim_lr=None):
        super().__init__()

        self.discrete = discrete
        self.action_dim = action_dim

        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        feature_dim = math.prod(repr_shape)

        trunk = nn.Sequential(nn.Linear(feature_dim, trunk_dim),
                              nn.LayerNorm(trunk_dim), nn.Tanh())

        self.trunk = Utils.default(call(recipe.trunk, feature_dim=feature_dim, trunk_dim=trunk_dim), trunk)

        out_dim = action_dim * 2 if stddev_schedule is None else action_dim

        self.Pi_head = Utils.Ensemble([MLP(trunk_dim, out_dim, hidden_dim, 2, l2_norm=l2_norm)
                                       for _ in range(ensemble_size)])

        # Pre-defined recipe, if provided
        if recipe.Pi_head is not None:
            kwargs = dict(trunk_dim=trunk_dim, out_dim=out_dim, hidden_dim=hidden_dim, l2_norm=l2_norm)
            self.Pi_head = Utils.Ensemble([call(recipe.Pi_head, **kwargs) for _ in range(ensemble_size)])

        self.init(optim_lr, ema_tau)

    def init(self, optim_lr=None, ema_tau=None):
        # Initialize weights
        self.apply(Utils.weight_init)

        # Optimizer
        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        # EMA
        if ema_tau is not None:
            self.ema = copy.deepcopy(self)
            self.ema_tau = ema_tau

    def update_ema_params(self):
        assert hasattr(self, 'ema')
        Utils.param_copy(self, self.ema, self.ema_tau)

    def forward(self, obs, step=None):
        h = self.trunk(obs)

        if self.stddev_schedule is None or step is None:
            mean_tanh, log_stddev = self.Pi_head(h).chunk(2, dim=-1)
            stddev = torch.exp(log_stddev)
        else:
            mean_tanh = self.Pi_head(h)
            stddev = torch.full_like(mean_tanh,
                                     Utils.schedule(self.stddev_schedule, step))

        self.mean_tanh = mean_tanh  # Pre-Tanh mean can be regularized (https://openreview.net/pdf?id=9xhgmsNVHu)
        mean = torch.tanh(self.mean_tanh)

        Pi = TruncatedNormal(mean, stddev, low=-1, high=1, stddev_clip=self.stddev_clip)

        return Pi


class CategoricalCriticActor(nn.Module):
    def __init__(self, entropy_sched=1):
        super().__init__()

        self.entropy_sched = entropy_sched

    def forward(self, Q, step=None, exploit_temp=1, sample_q=False, actions_log_prob=0):
        # Sample q or mean
        q = Q.rsample() if sample_q else Q.mean

        u = exploit_temp * q + (1 - exploit_temp) * Q.stddev
        u_logits = u - u.max(dim=-1, keepdim=True)[0]
        entropy_temp = Utils.schedule(self.entropy_sched, step)
        Q_Pi = Categorical(logits=u_logits / entropy_temp + actions_log_prob)

        best_eps, best_ind = torch.max(u, -1)
        best_action = Utils.gather_indices(Q.action, best_ind.unsqueeze(-1), 1).squeeze(1)

        sample = Q_Pi.sample

        def action_sampler(sample_shape=torch.Size()):
            i = sample(sample_shape)
            return Utils.gather_indices(Q.action, i.unsqueeze(-1), 1).squeeze(1)

        Q_Pi.__dict__.update({'best': best_action,
                              'best_u': best_eps,
                              'sample_ind': sample,
                              'sample': action_sampler,
                              'Q': Q,
                              'q': q,
                              'actions': Q.action,
                              'u': u})
        return Q_Pi
