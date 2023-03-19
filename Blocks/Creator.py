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
    Policy distribution and probabilistic measures for sampling across action spaces and ensembles.
    """
    def __init__(self, action_spec, critic=None, ActionExtractor=None, sample_discrete_as_discrete=True,
                 discrete=False, temp_schedule=1, stddev_clip=torch.inf, optim=None, scheduler=None,
                 lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.discrete = discrete
        self.action_dim = math.prod(action_spec.shape)  # d

        self.low, self.high = action_spec.low, action_spec.high

        # Entropy value, max cutoff clip for action sampling
        self.temp_schedule, self.stddev_clip = temp_schedule, stddev_clip

        # A mapping applied after sampling but prior to ensemble reduction
        self.ActionExtractor = Utils.instantiate(ActionExtractor, in_shape=self.action_dim) or nn.Identity()

        # Re-sample a discrete action from a sampled continuous categorical distribution
        self.sample_discrete_as_discrete = action_spec.discrete and not self.discrete and sample_discrete_as_discrete

        self.Pi = self.first_sample = self.action = self.best_action = self.step = self.obs = None
        self.critic = critic

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).requires_grad_(False)

    # Sets distribution
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

        self.best_action = None

        # Returns itself as policy
        return self

    def probs(self, action, judgement=None, as_ensemble=True):
        # Individual action probability
        probs = self.Pi.probs(action)

        # If action is an ensemble, multiply probability of sampling action from the ensemble
        if as_ensemble and not self.discrete and self.critic is not None and action.shape[1] > 1:
            if judgement is None:
                judgement, _ = self.critic.judgement(self.critic(self.obs, action))  # Reduced critic-ensemble Q-values

            if judgement is not None:
                judgement = judgement - judgement.max(-1, keepdim=True)[0]  # Normalize critic's judgement ensemble-wise

                # Categorical distribution based on critic's judgement
                temp = Utils.schedule(self.temp_schedule, self.step)  # Softmax temperature / "entropy"
                probs *= (judgement / temp).softmax(-1)  # Actor-ensemble-wise probabilities

        # Multiply probability of sampling a discrete action from a continuous-ized categorical distribution
        if self.sample_discrete_as_discrete:
            pass  # TODO - And has to be reshaped to n, d rather than nd before Softmax

        return probs

    # Exploration policy
    def sample(self, sample_shape=None, detach=True):
        # Sample
        action = self.ActionExtractor(self.Pi.sample(sample_shape or 1) if detach
                                      else self.Pi.rsample(sample_shape or 1))

        # Reduce Actor ensemble
        if self.critic is not None and sample_shape is None and action.shape[1] > 1 and not self.discrete:
            Qs = self.critic(self.obs, action)
            judgement, _ = self.critic.judgement(Qs)  # Reduce critic ensemble (e.g. Pessimism <--> Min-reduce)

            # Normalize
            judgement -= judgement.max(-1, keepdim=True)[0]

            # Softmax temperature
            temp = Utils.schedule(self.temp_schedule, self.step) if self.step else 1

            # Categorical dist
            Psi = torch.distributions.Categorical(logits=judgement / temp)

            # Sample again ensemble-wise
            action = Utils.gather(action, Psi.sample().unsqueeze(-1), 1).squeeze(1)

        # Can sample again from a continuous categorical distribution to get a discrete action if action continuous-ized
        if self.sample_discrete_as_discrete:
            pass  # TODO - And has to be reshaped to n, d rather than nd before Softmax?

        return action

    def rsample(self, sample_shape=None):
        return self.sample(sample_shape, detach=False)  # Differentiable explore

    # Exploitation policy
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
            q = Qs.mean(1)  # Mean-reduced ensemble. Note: Not using self.critic.judgement(Qs)

            # Normalize
            q -= q.max(-1, keepdim=True)[0]

            # Highest Q-value
            _, best_ind = q.max(-1)

            # Action corresponding to highest Q-value
            action = Utils.gather(action, best_ind.unsqueeze(-1), 1).squeeze(1)

        # Can Argmax again from a continuous categorical distribution to get a discrete action if action continuous-ized
        if self.sample_discrete_as_discrete:
            pass  # TODO - And has to be reshaped to n, d rather than nd before Argmax?

        self.best_action = action

        return action
