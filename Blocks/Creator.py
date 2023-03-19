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
    A distribution consisting of an exploration policy and an exploitation policy
    """
    def __init__(self, action_spec, critic=None, explore=None, exploit=None, second_sample=False,
                 discrete=False, temp_schedule=1, stddev_clip=torch.inf, optim=None, scheduler=None,
                 lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.action_dim = math.prod(action_spec.shape)  # d

        self.low, self.high = action_spec.low, action_spec.high

        self.second_sample = second_sample

        self.first_sample = None

        # Exploration / Exploitation policies
        self.Explore = MonteCarloPolicy(discrete, temp_schedule, stddev_clip, self.low, self.high, critic)
        self.Exploit = ArgmaxPolicy(discrete, self.low, self.high, critic)

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

    def probs(self, action, q=None, as_ensemble=True):
        return self.Explore.probs(action, q, as_ensemble)

    def sample(self, sample_shape=None, detach=True):
        action = self.first_sample = self.Explore.sample(sample_shape) if detach else self.Explore.rsample(sample_shape)

        if self.second_sample and not self.discrete:
            pass  # TODO Second sample

        return action

    def rsample(self, sample_shape=None):
        return self.sample(sample_shape, detach=False)

    @property
    def best(self):
        action = self.Exploit.sample()

        return action  # Best action


class MonteCarloPolicy(torch.nn.Module):
    """
    RL policy - samples action across action space and ensemble space

    A policy must contain a sample(·) method
            sample: sample_shape (default None) -> action(s)
                    If sample_shape is None, the policy returns exactly 1 action
                    Otherwise, the policy returns a factor N=prod(sample_shape) of the actor ensemble size actions
    For use in Q-learning, a policy should include a probs(·) method
            probs: action, q -> probability
    Any post-sampling, pre-ensemble-reduction operations may be specified by ActionExtractor
    """
    def __init__(self, discrete=True, temp_schedule=1, stddev_clip=None, low=None, high=None, critic=None):
        super().__init__()

        self.Pi = None

        self.discrete = discrete

        # Entropy value, max cutoff clip for action sampling
        self.temp_schedule, self.stddev_clip = temp_schedule, stddev_clip

        self.low, self.high = low, high

        self.step = self.obs = None
        self.critic = critic

    def forward(self, action, explore_rate, step, obs):
        if self.discrete:
            # Pessimism  TODO perhaps critic.judgement()
            logits, ind = action.min(1)  # Min-reduced ensemble [b, n, d]
            stddev = Utils.gather(explore_rate, ind.unsqueeze(1), 1, 1).squeeze(1)  # Min-reduced ensemble std [b, n, d]

            self.Pi = NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)
        else:
            self.Pi = TruncatedNormal(action, explore_rate, low=self.low, high=self.high, stddev_clip=self.stddev_clip)

        # For critic-based ensemble reduction
        if self.critic is not None:
            self.step = step
            self.obs = obs

    def probs(self, action, q=None, as_ensemble=True):
        probs = self.Pi.probs(action)

        if not as_ensemble:
            return probs

        # Ensembles are reduced based on Q-value
        if not (self.critic is None or self.discrete) and q is None and action.shape[1] > 1:
            # Pessimistic Q-values per action
            q = self.critic(self.obs, action).min(1)[0]  # Min-reduced critic ensemble

        # Ensemble-wise probability
        if q is not None:
            q_norm = q - q.max(-1, keepdim=True)[0]  # Normalized ensemble-wise

            # Categorical distribution based on Q-value
            temp = Utils.schedule(self.temp_schedule, self.step)  # Softmax temperature / "entropy"
            probs *= (q_norm / temp).softmax(-1)  # Probabilities

        return probs

    def sample(self, sample_shape=None, detach=True):
        action = self.Pi.sample(sample_shape or 1) if detach else self.Pi.rsample(sample_shape or 1)
        # action = self.ActionExtractor(action)  # TODO

        # Reduce ensemble
        if self.critic is not None and sample_shape is None and action.shape[1] > 1 and not self.discrete:
            # Pessimistic Q-values per action
            Qs = self.critic(self.obs, action)
            q = Qs.min(1)[0]  # Min-reduced critic ensemble

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


class ArgmaxPolicy(torch.nn.Module):
    """
    RL policy - samples action across action space and ensemble space

    A policy must contain a sample(·) method
            sample: sample_shape (default None) -> action(s)
                    If sample_shape is None, the policy returns exactly 1 action
                    Otherwise, the policy returns a factor N=prod(sample_shape) of the actor ensemble size actions
    For use in Q-learning, a policy should include a probs(·) method
            probs: action, q -> probability
    Any post-sampling, pre-ensemble-reduction operations may be specified by ActionExtractor
    """
    def __init__(self, discrete=True, low=None, high=None, critic=None):
        super().__init__()

        self.discrete = discrete
        self.low, self.high = low, high
        self.best = self.obs = None
        self.critic = critic

    def forward(self, action, explore_rate=None, step=None, obs=None):
        # Argmax for Discrete, Mean for Continuous
        self.best = action.argmax(-1, keepdim=True).transpose(-1, -2) if self.discrete else action

        # Normalize Discrete Action -> [low, high]
        if self.discrete and self.low is not None and self.high is not None:
            self.best = self.best / (action.shape[-1] - 1) * (self.high - self.low) + self.low

        # For critic-based ensemble reduction
        if self.critic is not None:
            self.obs = obs

    def probs(self, action, q=None, as_ensemble=True):
        # Absolute Determinism
        return torch.ones([1], device=action.device).expand(len(action), 1)

    def sample(self, sample_shape=None):
        action = self.best  # Note: sample_shape is at most (1,)
        # action = self.ActionExtractor(action)  # TODO

        # Reduce ensemble
        if self.critic is not None and sample_shape is None and action.shape[1] > 1 and not self.discrete:
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
