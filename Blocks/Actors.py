# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

import torch
from torch import nn

from Distributions import TruncatedNormal, NormalizedCategorical

from Blocks.Architectures.MLP import MLP

import Utils


class EnsemblePiActor(nn.Module):
    """Ensemble of Gaussian or Categorical policies Pi, generalized for discrete or continuous action spaces."""
    def __init__(self, repr_shape, trunk_dim, hidden_dim, action_spec, trunk=None, Pi_head=None, ActionExtractor=None,
                 ensemble_size=2, discrete=False, stddev_schedule=1, stddev_clip=torch.inf, optim=None, scheduler=None,
                 lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.num_actions = action_spec.discrete_bins or 1  # n
        self.action_dim = math.prod(action_spec.shape) * (1 if stddev_schedule else 2)  # d, or d * 2

        # Standard dev value
        self.stddev_schedule = stddev_schedule

        in_dim = math.prod(repr_shape)

        self.trunk = Utils.instantiate(trunk, input_shape=repr_shape, output_shape=[trunk_dim]) or nn.Sequential(
            nn.Flatten(), nn.Linear(in_dim, trunk_dim), nn.LayerNorm(trunk_dim), nn.Tanh())

        in_shape = Utils.cnn_feature_shape(repr_shape, self.trunk)  # Will be trunk_dim when possible
        out_shape = [self.num_actions * action_spec.shape[0] * (1 if stddev_schedule else 2), *action_spec.shape[1:]]

        # Ensemble
        self.Pi_head = Utils.Ensemble([Utils.instantiate(Pi_head, i, input_shape=in_shape, output_shape=out_shape)
                                       or MLP(in_shape, out_shape, hidden_dim, 2) for i in range(ensemble_size)])

        # Categorical policy for discrete, Normal for continuous
        self.dist = MonteCarloCreator(discrete, action_spec.low, action_spec.high, stddev_clip)

        # A mapping that can be applied after continuous-action sampling but prior to ensemble reduction
        self.ActionExtractor = Utils.instantiate(ActionExtractor, input_shape=self.action_dim) or nn.Identity()

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).requires_grad_(False)

    def forward(self, obs, step=1):
        h = self.trunk(obs)

        mean = self.Pi_head(h).view(h.shape[0], -1, self.num_actions, self.action_dim)  # [b, e, n, d or 2 * d]

        if self.stddev_schedule is None:
            mean, log_stddev = mean.chunk(2, dim=-1)  # [b, e, n, d]
            stddev = log_stddev.exp()  # [b, e, n, d]
        else:
            stddev = torch.full_like(mean, Utils.schedule(self.stddev_schedule, step))  # [b, e, n, d]

        Pi = self.dist(mean, stddev)

        # Secondary action extraction from samples
        if not self.discrete:
            sampler = Pi.sample

            def new_sampler(sample_shape=1):
                sample = sampler(sample_shape)

                return self.ActionExtractor(sample.view(*sample.shape[:-1], self.num_actions, -1)).view_as(sample)

            Pi.sample = new_sampler

        return Pi


class MonteCarloCreator(nn.Module):
    def __init__(self, discrete, low=None, high=None, stddev_clip=math.inf):
        super().__init__()

        self.discrete = discrete

        self.low, self.high = low, high

        # Max cutoff clip for action sampling
        self.stddev_clip = stddev_clip

    """Policy distribution for sampling across action spaces."""
    def forward(self, mean, stddev):
        if self.discrete:
            logits, ind = mean.min(1)  # Min-reduced ensemble [b, n, d]
            stddev = Utils.gather(stddev, ind.unsqueeze(1), 1, 1).squeeze(1)  # Min-reduced ensemble stand dev [b, n, d]

            Psi = NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)

            # All actions' Q-values
            setattr(Psi, 'All_Qs', mean)  # [b, e, n, d]
        else:
            if self.low or self.high:
                mean = (torch.tanh(mean) + 1) / 2 * (self.high - self.low) + self.low  # Normalize  [b, e, n, d]

            Psi = TruncatedNormal(mean, stddev, low=self.low, high=self.high, stddev_clip=self.stddev_clip)

        return Psi


class CategoricalCriticCreator(nn.Module):
    """Policy distribution that samples over ensembles and selects actions based on Q-values."""
    def __init__(self, temp_schedule=1):
        super().__init__()

        self.temp_schedule = temp_schedule

    def forward(self, Qs, step=None, action=None):
        # Q-values per action
        q = Qs.mean(1)  # Mean-reduced ensemble

        # Normalize
        q -= q.max(-1, keepdim=True)[0]

        # Softmax temperature
        temp = Utils.schedule(self.temp_schedule, step) if step else 1

        # Categorical dist
        Psi = torch.distributions.Categorical(logits=q / temp)

        # Highest Q-value
        _, best_ind = q.max(-1)

        # Action corresponding to highest Q-value
        setattr(Psi, 'best', best_ind if action is None else Utils.gather(action, best_ind.unsqueeze(-1), 1).squeeze(1))

        # Action sampling
        sampler = Psi.sample
        Psi.sample = sampler if action is None else lambda: Utils.gather(action, sampler().unsqueeze(-1), 1).squeeze(1)

        return Psi
