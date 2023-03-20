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
    def __init__(self, action_spec, policy=None, ActionExtractor=None,
                 optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.policy = policy  # Exploration and exploitation policy recipe

        self.action_spec = action_spec

        # A mapping that can be applied after or concurrently with action sampling
        self.ActionExtractor = Utils.instantiate(ActionExtractor, input_shape=math.prod(self.action_spec.shape)
                                                 ) or nn.Identity()

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).requires_grad_(False)

    def Pi(self, mean, stddev, discrete=False, stddev_clip=torch.inf):
        # Return policy distribution
        return Utils.instantiate(self.policy, mean=mean, stddev=stddev, action_spec=self.action_spec,
                                 ActionExtractor=self.ActionExtractor, discrete=discrete, stddev_clip=stddev_clip) or \
            MonteCarlo(mean, stddev, self.action_spec, self.ActionExtractor, discrete, stddev_clip)


# Policy
class MonteCarlo(torch.nn.Module):
    """Exploration and exploitation policy distribution compatible with discrete and continuous spaces and ensembles."""
    def __init__(self, mean, stddev, action_spec, ActionExtractor=None, discrete=False, stddev_clip=torch.inf):
        super().__init__()

        self.discrete = discrete

        self.low, self.high = action_spec.low, action_spec.high

        # Policy
        if self.discrete:
            # Pessimistic Q-values per action
            logits, ind = self.action.min(1)  # Min-reduced ensemble [b, n, d]
            stddev = Utils.gather(stddev, ind.unsqueeze(1), 1, 1).squeeze(1)  # Min-reduced ensemble std [b, n, d]

            self.Psi = NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)
            self.All_Qs = mean  # [b, e, n, d]
            self.logits = self.Psi.logits
        else:
            self.Psi = TruncatedNormal(self.action, stddev, low=self.low, high=self.high, stddev_clip=stddev_clip)
            self.mean = mean  # [b, e, n, d]

        self.ActionExtractor = ActionExtractor

    def log_prob(self, action):
        # Individual action probability
        log_prob = self.Psi.log_prob(action)  # (Log-space is more numerically stable)

        # TODO 2nd sample

        return log_prob

    def entropy(self, action):
        pass  # TODO

    # Exploration policy
    def sample(self, sample_shape=None, detach=True):
        # Monte Carlo

        # Sample
        action = self.Psi.sample(sample_shape or 1) if detach else self.Psi.rsample(sample_shape or 1)

        if not self.discrete:
            action = self.ActionExtractor(action)

        if sample_shape is None and action.shape[1] > 1 and not self.discrete:
            pass  # TODO Uniform sample (index via rand int)

        # TODO 2nd sample

        return action

    def rsample(self, sample_shape=None):
        return self.sample(sample_shape, detach=False)  # Differentiable explore

    # Exploitation policy
    @cached_property
    def best(self):
        # Absolute Determinism

        # Argmax for discrete, extract action for continuous
        return self.Psi.normalize(self.logits.argmax(-1, keepdim=True).transpose(-1, -2)) if self.discrete \
            else self.ActionExtractor(self.mean)
