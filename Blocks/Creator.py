# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
from functools import cached_property
import copy

import torch
from torch import nn

from Distributions import TruncatedNormal, NormalizedCategorical

import Utils


class Creator(torch.nn.Module):
    """Creates a policy distribution for sampling actions and ensembles and computing probabilistic measures."""
    def __init__(self, action_spec, policy=None, ActionExtractor=None, discrete=False, stddev_clip=torch.inf,
                 optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.discrete = discrete
        self.action_spec = action_spec

        # Max cutoff clip for continuous-action sampling
        self.stddev_clip = stddev_clip  # TODO Make this default/automatic. Include kwargs maybe for below.

        self.policy = policy  # Exploration and exploitation policy recipe

        # A mapping that can be applied after or concurrently with action sampling
        self.ActionExtractor = Utils.instantiate(ActionExtractor, input_shape=math.prod(self.action_spec.shape)
                                                 ) or nn.Identity()

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).requires_grad_(False)

    # Creates actor policy Pi
    def Omega(self, mean, stddev, step=1):  # TODO Rename maybe (act, explore)
        # Optionally create policy from recipe
        return Utils.instantiate(self.policy, mean=mean, stddev=stddev, action_spec=self.action_spec, step=step,
                                 ActionExtractor=self.ActionExtractor, discrete=self.discrete,
                                 stddev_clip=self.stddev_clip) or \
            MonteCarlo(mean, stddev, self.action_spec, self.ActionExtractor, self.discrete, self.stddev_clip)  # Default


# Policy
class MonteCarlo(torch.nn.Module):
    """Exploration and exploitation policy distribution compatible with discrete and continuous spaces and ensembles."""
    def __init__(self, mean, stddev, action_spec, ActionExtractor=None, discrete=False, stddev_clip=torch.inf):
        super().__init__()

        self.discrete = discrete
        self.discrete_as_continuous = action_spec.discrete and not self.discrete

        self.low, self.high = action_spec.low, action_spec.high

        # Max cutoff clip for action sampling
        self.stddev_clip = stddev_clip

        # Policy
        if self.discrete:
            self.All_Qs = mean  # [b, e, n, d]
        else:
            self.mean = mean  # [b, e, n, d]

        self.stddev = stddev

        self.ActionExtractor = ActionExtractor

    #     self.critic = None  TODO
    #
    # # Can enable critic-based ensemble reduction (optional)
    # def forward(self, critic):
    #     # This policy can accept a critic for non-random ensemble reduction
    #     self.critic = critic
    #
    #     # Returns itself
    #     return self

    # Policy
    @cached_property
    def Psi(self):
        if self.discrete:
            # Pessimistic Q-values per action
            logits, ind = self.All_Qs.min(1)  # Min-reduced ensemble [b, n, d]
            stddev = Utils.gather(self.stddev, ind.unsqueeze(1), 1, 1).squeeze(1)  # Min-reduced ensemble std [b, n, d]

            try:
                return NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)
            except ValueError:
                # M1 Mac MPS can sometimes throw ValueError on log-sum-exp trick
                Psi = NormalizedCategorical(logits=logits.to('cpu'), low=self.low, high=self.high, temp=stddev, dim=-2)
                Psi.logits = Psi.logits.to(logits.device)
                return Psi
        else:
            return TruncatedNormal(self.mean, self.stddev, low=self.low, high=self.high, stddev_clip=self.stddev_clip)

    def log_prob(self, action):
        # Log-probability
        log_prob = self.Psi.log_prob(action)  # (Log-space is more numerically stable)  TODO critic

        # If continuous-action is a discrete distribution, it gets double-sampled
        if self.discrete_as_continuous:
            log_prob += action  # (Adding log-probs is equal to multiplying probs)  TODO Temp

        return log_prob  # [b, e] TODO Might need to align dims

    @cached_property
    def _entropy(self):
        return self.Psi.entropy() if self.discrete else self.Psi.entropy().mean(-1)  # [b, e]

    def entropy(self, action=None):
        # If continuous-action is a discrete distribution, 2nd sample also has entropy
        if self.discrete_as_continuous:
            # Approximate joint entropy  TODO Might need to align dims
            return self._entropy + torch.distributions.Categorical(logits=action).entropy()  # TODO Temp

        return self._entropy  # [b, e]  TODO critic

    # Exploration policy
    def sample(self, sample_shape=None, detach=True):
        # Monte Carlo

        # Sample
        action = self.Psi.sample(sample_shape or 1) if detach else self.Psi.rsample(sample_shape or 1)

        # Can optionally map a continuous-action after sampling
        if not self.discrete:
            action = self.ActionExtractor(action)

        # Reduce continuous-action ensemble
        if sample_shape is None and action.shape[1] > 1 and not self.discrete:
            return action[:, torch.randint(action.shape[1], [])]  # Uniform sample again across ensemble  TODO critic

        # If sampled action is a discrete distribution, sample again
        if self.discrete_as_continuous:
            action = torch.distributions.Categorical(logits=action).sample()  # Sample again  TODO Temp

        return action

    def rsample(self, sample_shape=None):
        return self.sample(sample_shape, detach=False)  # Differentiable explore

    # Exploitation policy
    @cached_property
    def best(self):
        # Absolute Determinism

        # Argmax for discrete, extract action for continuous  Note: Ensembles somewhat random; should use critic
        action = self.Psi.normalize(self.Psi.logits.argmax(-1, keepdim=True).transpose(-1, -2)) if self.discrete \
            else self.ActionExtractor(self.mean[:, torch.randint(self.mean.shape[1], [])])  # TODO critic

        # If continuous-action is a discrete distribution
        if self.discrete_as_continuous:
            action = action.argmax(-1)  # Argmax

        return action
