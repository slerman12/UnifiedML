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
                 discrete=False, temp_schedule=1, stddev_clip=torch.inf, optim=None, scheduler=None,
                 lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.action_dim = math.prod(action_spec.shape)  # d

        self.low, self.high = action_spec.low, action_spec.high

        self.second_sample = second_sample

        self.action = self.first_sample = None

        # Exploration / Exploitation policies
        self.Explore = MonteCarloPolicy(discrete, temp_schedule, stddev_clip, self.low, self.high, critic)
        self.Exploit = BestPolicy(discrete, critic)

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).requires_grad_(False)

    def forward(self, action, explore_rate, step):
        # Initialize policies
        self.Explore(action, explore_rate, step)
        self.Exploit(action)

        return self  # Return self as an initialized distribution

    def probs(self, action, Qs=None):
        return self.Explore.probs(action, Qs) if hasattr(self.Explore, 'probs') \
            else torch.ones(1, device=action.device)

    def sample(self, sample_shape=None, detach=True):
        sample = self.Explore.sample(sample_shape) if detach else self.Explore.rsample(sample_shape)

        # Normalize Discrete Action -> [low, high]
        if self.discrete and self.low is not None and self.high is not None:
            sample = sample / (self.action_dim - 1) * (self.high - self.low) + self.low

        self.first_sample = sample

        if self.second_sample and not self.discrete:
            pass  # TODO Second sample

        return sample

    def rsample(self, sample_shape=None):
        return self.sample(sample_shape, detach=False)

    @property
    def best(self):
        action = self.Exploit.sample()

        # Normalize Discrete Action -> [low, high]
        if self.discrete and self.low is not None and self.high is not None:
            action = action / (self.action_dim - 1) * (self.high - self.low) + self.low

        return action  # Best action


class MonteCarloPolicy(torch.nn.Module):
    """
    RL policy - samples action across action space and ensemble space

    A policy must contain a sample(·) method
            sample: sample_shape (default None) -> action(s)
                    If sample_shape is None, the policy returns exactly 1 action
                    Otherwise, the policy returns a factor N=prod(sample_shape) of the actor ensemble size actions
    For use in Q-learning, a policy should include a probs(·) method
            probs: action, Qs -> probability
    """
    def __init__(self, discrete, temp_schedule, stddev_clip, low, high, critic):
        super().__init__()

        self.Pi = None

        self.discrete = discrete

        # Entropy value, max cutoff clip for action sampling
        self.temp_schedule, self.stddev_clip = temp_schedule, stddev_clip

        self.low, self.high = low, high

        self.step = self.obs = None
        self.critic = critic

    def forward(self, mean, stddev, step, obs):
        if self.discrete:
            # Pessimism
            logits, ind = mean.min(1)  # Min-reduced ensemble [b, n, d]
            stddev = Utils.gather(stddev, ind.unsqueeze(1), 1, 1).squeeze(1)  # Min-reduced ensemble stand dev [b, n, d]

            self.Pi = NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)
        else:
            self.Pi = TruncatedNormal(mean, stddev, low=self.low, high=self.high, stddev_clip=self.stddev_clip)

        # For critic-based ensemble reduction
        self.step = step
        self.obs = obs

    def probs(self, action, Qs=None):
        pass  # TODO

    def sample(self, sample_shape=None, detach=True):
        action = self.Pi.sample(sample_shape or 1) if detach else self.Pi.rsample(sample_shape or 1)
        # action = self.ActionExtractor(action)  # TODO

        # Reduce ensemble
        if sample_shape is None and action.shape[1] > 1 and not self.discrete:
            # Q-values per action
            Qs = self.critic(self.obs, action)
            q = Qs.mean(1)  # Mean-reduced ensemble

            # Normalize
            q -= q.max(-1, keepdim=True)[0]

            # Softmax temperature
            temp = Utils.schedule(self.temp_schedule, self.step) if self.step else 1

            # Categorical dist
            Psi = torch.distributions.Categorical(logits=q / temp)

            action = Utils.gather(action, Psi.sample().unsqueeze(-1), 1).squeeze(1)

        return action

    def rsample(self, sample_shape=None):
        return self.sample(sample_shape, detach=False)


class BestPolicy(torch.nn.Module):
    """
    RL policy - samples action across action space and ensemble space

    A policy must contain a sample(·) method
            sample: sample_shape (default None) -> action(s)
                    If sample_shape is None, the policy returns exactly 1 action
                    Otherwise, the policy returns a factor N=prod(sample_shape) of the actor ensemble size actions
    For use in Q-learning, a policy should include a probs(·) method
            probs: action, Qs -> probability
    """
    def __init__(self, discrete, critic):
        super().__init__()

        self.best = None
        self.discrete = discrete
        self.obs = None
        self.critic = critic

    def forward(self, action, explore_rate=None, step=None, obs=None):
        # Argmax for Discrete, Mean for Continuous
        self.best = action.argmax(-1, keepdim=True).transpose(-1, -2) if self.discrete else action

        # For critic-based ensemble reduction
        self.obs = obs

    def sample(self, sample_shape=None):
        action = self.best  # Note: sample_shape is at most (1,)

        # Reduce ensemble
        if sample_shape is None and action.shape[1] > 1 and not self.discrete:
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

    def rsample(self, sample_shape=None):
        return self.sample(sample_shape)  # Note: Not differentiable
