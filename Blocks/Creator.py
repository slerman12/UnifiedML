# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

from Distributions import TruncatedNormal, NormalizedCategorical

import Utils


class MonteCarloCreator(torch.nn.Module):
    """
    Selects over actions based on probabilities and "goodness"
    """
    def __init__(self, action_spec, critic, discrete=True, stddev_schedule=1, stddev_clip=torch.inf):
        super().__init__()

        self.low, self.high = action_spec.low, action_spec.high

        self.Explore = MonteCarloPolicy(discrete, stddev_schedule, stddev_clip, self.low, self.high, critic)
        self.Exploit = BestPolicy(discrete, critic)

    def sample(self, sample_shape=1):
        sample = self.Explore.sample(sample_shape)

        # Normalize Discrete Action -> [low, high]
        if self.discrete:
            sample = self.normalize(sample)

        return sample

    @property
    def best(self):
        sample = self.Exploit.sample()

        # Normalize Discrete Action -> [low, high]
        if self.discrete:
            sample = self.normalize(sample)

        return sample

    def goodness(self, action):
        return self.Exploit.goodness(action)

    def forward(self, action, explore_rate):
        self.Explore(action, explore_rate)
        self.Exploit(action)

        return self

    def normalize(self, sample):
        # Normalize Discrete Sample -> [low, high]  TODO logits shape
        if self.low is not None and self.high is not None:
            sample = sample / (self.logits.shape[-1] - 1) * (self.high - self.low) + self.low

        return sample


class MonteCarloPolicy(torch.nn.Module):
    def __init__(self, discrete, stddev_schedule, stddev_clip, low, high, critic):
        super().__init__()

        self.Pi = None

        self.discrete = discrete

        # Standard dev value, max cutoff clip for action sampling
        self.stddev_schedule, self.stddev_clip = stddev_schedule, stddev_clip

        self.low, self.high = low, high

        self.critic = critic  # TODO

    def goodness(self, action):
        pass

    def forward(self, mean, stddev):
        if self.discrete:
            logits, ind = mean.min(1)  # Min-reduced ensemble [b, n, d]
            stddev = Utils.gather(stddev, ind.unsqueeze(1), 1, 1).squeeze(1)  # Min-reduced ensemble stand dev [b, n, d]

            self.Pi = NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)
        else:
            self.Pi = TruncatedNormal(mean, stddev, low=self.low, high=self.high, stddev_clip=self.stddev_clip)

    def sample(self, sample_shape=1):
        sample = self.Pi.sample(sample_shape)

        return sample


class BestPolicy(torch.nn.Module):
    def __init__(self, discrete, critic):
        super().__init__()

        self.best = None
        self.discrete = discrete
        self._goodness = torch.ones(1)
        self.critic = critic  # TODO

    def goodness(self, action):
        return self._goodness.expand(len(action))  # ensemble dim?

    def forward(self, actions):
        # Argmax for Discrete, Mean for Continuous
        self.best = actions.argmax(-1, keepdim=True).transpose(-1, -2) if self.discrete else actions

    def sample(self, sample_shape=1):
        return self.best  # Note: returns full ensemble!
