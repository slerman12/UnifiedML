# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

import torch
from torch import nn

from Distributions import TruncatedNormal, NormalizedCategorical

import Utils


class Creator(torch.nn.Module):
    """Creates a policy distribution for sampling actions and computing probabilistic measures."""
    def __init__(self, action_spec, ActionExtractor=None, discrete=False, temp_schedule=1, stddev_clip=math.inf,
                 optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.discrete = discrete
        self.action_dim = math.prod(action_spec.shape)  # d

        self.low, self.high = action_spec.low, action_spec.high

        # Entropy value for ensemble reduction, max cutoff clip for action sampling
        self.temp_schedule, self.stddev_clip = temp_schedule, stddev_clip

        self.Dist = ExploreExploit  # Exploration and exploitation

        # A mapping applied after sampling but prior to ensemble reduction
        self.ActionExtractor = Utils.instantiate(ActionExtractor, in_shape=self.action_dim) or nn.Identity()

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).requires_grad_(False)

        self.critic = None  # Critic can be used to reduce continuous-action ensembles

    def forward(self, critic):
        self.critic = critic  # Enable critic-based ensemble reduction
        return self

    # Get policy
    def dist(self, action, explore_rate, step=1):
        return self.Dist(action, explore_rate, step, self.ActionExtractor, self.critic)


class ExploreExploit(torch.nn.Module):
    """Exploration and exploitation distribution compatible with discrete and continuous spaces and ensembles."""
    def __init__(self, action, explore_rate, step=1, ActionExtractor=None, critic=None):
        super().__init__()

        self.action = action  # [b, e, n, d]
        self.step = step

        if self.discrete:
            logits, ind = action.min(1)  # Reduced ensemble [b, n, d]
            stddev = Utils.gather(explore_rate, ind.unsqueeze(1), 1, 1).squeeze(1)  # Reduced ensemble stddev [b, n, d]

            self.Pi = NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)
        else:
            self.Pi = TruncatedNormal(action, explore_rate, low=self.low, high=self.high, stddev_clip=self.stddev_clip)

        self.ActionExtractor = ActionExtractor
        self.critic = critic

        self._best = None

    def log_prob(self, action, q=None, as_ensemble=True):
        # Individual action probability
        log_prob = self.Pi.log_prob(action)  # (Log-space is more numerically stable)

        # If action is an ensemble, multiply probability of sampling action from the ensemble
        if as_ensemble and self.critic is not None and not self.discrete and action.shape[1] > 1:
            if q is None:
                Qs = self.critic(self.obs, action)
                q, _ = Qs.min(1)  # Reduced critic-ensemble Q-values

            q = q - q.max(-1, keepdim=True)[0]  # Normalize q ensemble-wise

            # Categorical distribution based on q
            temp = Utils.schedule(self.temp_schedule, self.step)  # Softmax temperature / "entropy"
            log_prob += q / temp  # Actor-ensemble-wise probabilities (Adding log-probs equivalent to multiplying probs)

        return log_prob

    # Exploration policy
    def sample(self, sample_shape=None, detach=True):
        # Monte Carlo

        # Sample
        action = self.Pi.sample(sample_shape or 1) if detach else self.Pi.rsample(sample_shape or 1)

        if not self.discrete:
            action = self.ActionExtractor(action)

        # Reduce Actor ensemble
        if sample_shape is None and self.critic is not None and action.shape[1] > 1 and not self.discrete:
            Qs = self.critic(self.obs, action)
            q, _ = Qs.min(1)  # Reduce critic ensemble (pessimism <-> Min-reduce)

            # Normalize
            q -= q.max(-1, keepdim=True)[0]

            # Softmax temperature
            temp = Utils.schedule(self.temp_schedule, self.step) if self.step else 1

            # Categorical dist
            Psi = torch.distributions.Categorical(logits=q / temp)

            # Sample again ensemble-wise
            action = Utils.gather(action, Psi.sample().unsqueeze(-1), 1).squeeze(1)

        return action

    def rsample(self, sample_shape=None):
        return self.sample(sample_shape, detach=False)  # Differentiable explore

    # Exploitation policy
    @property
    def best(self, sample_shape=None):
        # Absolute Determinism

        if self._best is None:
            # Argmax for discrete, extract action for continuous
            action = self.Pi.normalize(self.logits.argmax(-1, keepdim=True).transpose(-1, self.dim)) if self.discrete \
                else self.ActionExtractor(self.action)

            # Reduce ensemble
            if sample_shape is None and self.critic is not None and not self.discrete and action.shape[1] > 1:
                # Q-values per action
                Qs = self.critic(self.obs, action)
                q = Qs.mean(1)  # Mean-reduced ensemble. Note: Not using pessimism for best

                # Normalize
                q -= q.max(-1, keepdim=True)[0]

                # Highest Q-value
                _, best_ind = q.max(-1)

                # Action corresponding to highest Q-value
                action = Utils.gather(action, best_ind.unsqueeze(-1), 1).squeeze(1)

            self._best = action

        return self.best_action
