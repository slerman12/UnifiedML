# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

import torch

from Distributions import TruncatedNormal, NormalizedCategorical

import Utils


class Creator(torch.nn.Module):
    """
    Selects over action spaces based on probabilities and over ensembles based on Critic-evaluated "goodness"
    """
    def __init__(self, action_spec, critic, explore=None, exploit=None, second_sample=False,
                 discrete=False, stddev_schedule=1, stddev_clip=torch.inf, optim=None, scheduler=None,
                 lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.action_dim = math.prod(action_spec.shape)  # d

        self.low, self.high = action_spec.low, action_spec.high

        self.second_sample = second_sample

        self.action = self.first_sample = None

        # Exploration / Exploitation policies
        self.Explore = MonteCarloPolicy(discrete, stddev_schedule, stddev_clip, self.low, self.high, critic)
        self.Exploit = BestPolicy(discrete, critic)

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).requires_grad_(False)

    def forward(self, action, explore_rate):
        # Initialize policies
        self.Explore(action, explore_rate)
        self.Exploit(action)

        return self  # Return self as an initialized distribution

    def probs(self, action, Qs=None):
        return self.Explore.probs(action, Qs) if hasattr(self.Explore, 'probs') \
            else torch.ones(1, device=action.device)

    def sample(self, sample_shape=None):
        sample = self.Explore.sample(sample_shape)

        # Normalize Discrete Action -> [low, high]
        if self.discrete and self.low is not None and self.high is not None:
            sample = sample / (self.action_dim - 1) * (self.high - self.low) + self.low

        return sample

    @property
    def best(self):
        action = self.Exploit.sample()

        # Normalize Discrete Action -> [low, high]
        if self.discrete and self.low is not None and self.high is not None:
            action = action / (self.action_dim - 1) * (self.high - self.low) + self.low

        return action  # Best action


class MonteCarloPolicy(torch.nn.Module):
    """
    RL policy

    A policy must contain a sample(·) method
            sample: sample_shape -> action(s)
                    If sample_shape is None, the policy returns exactly 1 action
                    Otherwise, the policy returns a factor N of the action ensemble size actions
    For use in Q-learning, a policy should include a probs(·) method
            probs: action, Qs -> probability
    """
    def __init__(self, discrete, stddev_schedule, stddev_clip, low, high, critic):
        super().__init__()

        self.Pi = None

        self.discrete = discrete

        # Standard dev value, max cutoff clip for action sampling
        self.stddev_schedule, self.stddev_clip = stddev_schedule, stddev_clip

        self.low, self.high = low, high

        self.critic = critic

    def forward(self, mean, stddev):
        if self.discrete:  # TODO Maybe return these, pass all probs/sampling/even Qs to Distributions
            logits, ind = mean.min(1)  # Min-reduced ensemble [b, n, d]
            stddev = Utils.gather(stddev, ind.unsqueeze(1), 1, 1).squeeze(1)  # Min-reduced ensemble stand dev [b, n, d]

            self.Pi = NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)
        else:
            self.Pi = TruncatedNormal(mean, stddev, low=self.low, high=self.high, stddev_clip=self.stddev_clip)

    def probs(self, action, Qs=None):
        pass

    def sample(self, sample_shape=None):  # TODO rsample!!
        action = self.Pi.sample(sample_shape or 1)  # TODO critic if ensemble > 1 and sample_shape is None

        return action


class BestPolicy(torch.nn.Module):
    """
    RL policy

    A policy must contain a sample(·) method
            sample: sample_shape -> action(s)
                    If sample_shape is None, the policy returns exactly 1 action
                    Otherwise, the policy returns a factor N of the action ensemble size actions
    For use in Q-learning, a policy should include a probs(·) method
            probs: action, Qs -> probability
    """
    def __init__(self, discrete, critic):
        super().__init__()

        self.best = None
        self.discrete = discrete
        self.critic = critic

    def forward(self, actions):
        # Argmax for Discrete, Mean for Continuous
        self.best = actions.argmax(-1, keepdim=True).transpose(-1, -2) if self.discrete else actions

    def sample(self, sample_shape=None):
        return self.best  # TODO critic if ensemble > 1 and sample_shape is None
