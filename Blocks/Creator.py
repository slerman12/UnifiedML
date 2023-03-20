# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch

from Distributions import TruncatedNormal, NormalizedCategorical

import Utils


# TODO Outer Creator with Action Extractor, nothing else, initiates args entirely in forward or dist
class Creator(torch.nn.Module):
    """Exploration and exploitation policy distribution compatible with discrete and continuous spaces and ensembles."""
    def __init__(self, action, explore_rate, action_spec, discrete=False, stddev_clip=math.inf):
        super().__init__()

        self.discrete = discrete

        self.low, self.high = action_spec.low, action_spec.high

        # Policy
        if self.discrete:
            # Pessimistic Q-values per action
            logits, ind = self.action.min(1)  # Min-reduced ensemble [b, n, d]
            stddev = Utils.gather(explore_rate, ind.unsqueeze(1), 1, 1).squeeze(1)  # Min-reduced ensemble std [b, n, d]

            self.Psi = NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)

            self.All_Qs = action  # [b, e, n, d]

            self._best = None
        else:
            self.Psi = TruncatedNormal(self.action, explore_rate, low=self.low, high=self.high, stddev_clip=stddev_clip)

            self.mean = action  # [b, e, n, d]

    def log_prob(self, action):
        # Individual action probability
        log_prob = self.Psi.log_prob(action)  # (Log-space is more numerically stable)

        # TODO 2nd sample.

        return log_prob

    # Exploration policy
    def sample(self, sample_shape=None, detach=True):
        # Monte Carlo

        # Sample
        action = self.Psi.sample(sample_shape or 1) if detach else self.Psi.rsample(sample_shape or 1)

        if sample_shape is None and action.shape[1] > 1 and not self.discrete:
            pass  # TODO If critic None, uniform sample (index via rand int). Also, 2nd sample.

        return action

    def rsample(self, sample_shape=None):
        return self.sample(sample_shape, detach=False)  # Differentiable explore

    # Exploitation policy
    @property
    def best(self):
        # Absolute Determinism

        assert self.discrete, '"best" is only supported for discrete distributions. Try "mean" for continuous.'

        if self._best is None:
            # Argmax
            self._best = self.Psi.normalize(self.logits.argmax(-1, keepdim=True).transpose(-1, self.dim))

        return self._best
