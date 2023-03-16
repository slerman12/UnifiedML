# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

from Distributions import TruncatedNormal, NormalizedCategorical

import Utils


class MonteCarloCreator(torch.nn.Module):
    def __init__(self, action_spec, discrete=True, stddev_schedule=1, stddev_clip=torch.inf):
        super().__init__()

        self.low, self.high = action_spec.low, action_spec.high

        self.Explore = MonteCarloPolicy(discrete, stddev_schedule, stddev_clip, self.low, self.high)
        self.Exploit = BestPolicy(discrete, self.low, self.high)

    def sample(self):
        return self.Explore.sample()

    @property
    def best(self):
        return self.Exploit.sample()

    def goodness(self, action):
        return self.Exploit.goodness(action)

    def forward(self, actions, explore_rate):
        self.Explore(actions, explore_rate)
        self.Exploit(actions)

        return self


class MonteCarloPolicy(torch.nn.Module):
    def __init__(self, discrete, stddev_schedule, stddev_clip, low, high):
        super().__init__()

        self.Pi = None

        self.discrete = discrete

        # Standard dev value, max cutoff clip for action sampling
        self.stddev_schedule, self.stddev_clip = stddev_schedule, stddev_clip

        self.low, self.high = low, high

    def goodness(self, action):
        pass

    def forward(self, mean, stddev):
        if self.discrete:
            logits, ind = mean.min(1)  # Min-reduced ensemble [b, n, d]
            stddev = Utils.gather(stddev, ind.unsqueeze(1), 1, 1).squeeze(1)  # Min-reduced ensemble stand dev [b, n, d]

            self.Pi = NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)

            # All actions' Q-values
            # setattr(Pi, 'All_Qs', mean)  # [b, e, n, d]  # TODO Actor must store this
        else:
            if self.low or self.high:
                mean = (torch.tanh(mean) + 1) / 2 * (self.high - self.low) + self.low  # Normalize  [b, e, n, d]

            self.Pi = TruncatedNormal(mean, stddev, low=self.low, high=self.high, stddev_clip=self.stddev_clip)

    def sample(self):
        return self.Pi.sample()


class BestPolicy(torch.nn.Module):
    def __init__(self, discrete, stddev_schedule, stddev_clip, low, high):
        super().__init__()

        self.best = None

        self.discrete = discrete

        # Standard dev value, max cutoff clip for action sampling
        self.stddev_schedule, self.stddev_clip = stddev_schedule, stddev_clip

        self.low, self.high = low, high

    def goodness(self, action):
        pass

    def forward(self, mean):
        if self.discrete:
            self.best = self.normalize(mean.argmax(-1, keepdim=True).transpose(-1, self.dim))
        else:
            if self.low or self.high:
                mean = (torch.tanh(mean) + 1) / 2 * (self.high - self.low) + self.low  # Normalize  [b, e, n, d]

            self.best = mean

    def sample(self):
        return self.best
