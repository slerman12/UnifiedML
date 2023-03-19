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
    """
    Selects over action spaces based on probabilities and over ensembles based on Critic-evaluated "goodness"
    A distribution consisting of an exploration policy and an exploitation policy
    """
    def __init__(self, action_spec, critic=None, ActionExtractor=None, second_sample=False,
                 discrete=False, temp_schedule=1, stddev_clip=torch.inf, optim=None, scheduler=None,
                 lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.discrete = discrete
        self.action_dim = math.prod(action_spec.shape)  # d

        self.low, self.high = action_spec.low, action_spec.high

        # Entropy value, max cutoff clip for action sampling
        self.temp_schedule, self.stddev_clip = temp_schedule, stddev_clip

        self.ActionExtractor = Utils.instantiate(ActionExtractor) or nn.Identity()

        self.second_sample = second_sample

        self.Pi = self.first_sample = self.action = self.step = self.obs = None
        self.critic = critic

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).requires_grad_(False)

    def forward(self, action, explore_rate, step, obs):
        self.action = action

        if self.discrete:
            logits, ind = self.critic.judgement(action)  # Reduced ensemble [b, n, d]
            stddev = Utils.gather(explore_rate, ind.unsqueeze(1), 1, 1).squeeze(1)  # Reduced ensemble stddev [b, n, d]

            self.Pi = NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)
        else:
            self.Pi = TruncatedNormal(action, explore_rate, low=self.low, high=self.high, stddev_clip=self.stddev_clip)

        # For critic-based ensemble reduction
        if self.critic is not None:
            self.step = step
            self.obs = obs

    def probs(self, action, q=None, as_ensemble=True):
        probs = self.Pi.probs(action)

        # Ensemble-wise probabilities if actions are additionally sampled from ensembles
        if as_ensemble and not self.discrete and self.critic is not None and action.shape[1] > 1:
            if q is None:
                q, _ = self.critic.judgement(self.critic(self.obs, action))  # Reduced critic-ensemble Q-values

            if q is not None:
                q_norm = q - q.max(-1, keepdim=True)[0]  # Normalize ensemble-wise

                # Categorical distribution based on Q-value
                temp = Utils.schedule(self.temp_schedule, self.step)  # Softmax temperature / "entropy"
                probs *= (q_norm / temp).softmax(-1)  # Actor-ensemble-wise probabilities

        return probs

    # Explore
    def sample(self, sample_shape=None, detach=True):
        action = self.ActionExtractor(self.Pi.sample(sample_shape or 1) if detach
                                      else self.Pi.rsample(sample_shape or 1))

        # Reduce ensemble
        if self.critic is not None and sample_shape is None and action.shape[1] > 1 and not self.discrete:
            Qs = self.critic(self.obs, action)
            q, _ = self.critic.judgement(Qs)  # Reduce ensemble

            # Normalize
            q -= q.max(-1, keepdim=True)[0]

            # Softmax temperature
            temp = Utils.schedule(self.temp_schedule, self.step) if self.step else 1

            # Categorical dist
            Psi = torch.distributions.Categorical(logits=q / temp)

            action = Utils.gather(action, Psi.sample().unsqueeze(-1), 1).squeeze(1)

        return action

    # Differentiable Explore
    def rsample(self, sample_shape=None):
        return self.sample(sample_shape, detach=False)

    # Exploit
    @property
    def best(self):
        # Absolute Determinism

        # Argmax for Discrete, Mean for Continuous
        action = self.action.argmax(-1, keepdim=True).transpose(-1, -2) if self.discrete else self.action

        # Normalize Discrete Action -> [low, high]
        if self.discrete and self.low is not None and self.high is not None:
            action = action / (self.action.shape[-1] - 1) * (self.high - self.low) + self.low

        action = self.ActionExtractor(action)

        # Reduce ensemble
        if self.critic is not None and not self.discrete and action.shape[1] > 1:
            # Q-values per action
            Qs = self.critic(self.obs, action)
            q = Qs.mean(1)  # Mean-reduced ensemble

            # Normalize
            q -= q.max(-1, keepdim=True)[0]

            # Highest Q-value
            _, best_ind = q.max(-1)

            # Action corresponding to highest Q-value
            action = Utils.gather(action, best_ind.unsqueeze(-1), 1).squeeze(1)

        return action
