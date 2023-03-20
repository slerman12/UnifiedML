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


class Creator(nn.Module):
    """Creates a policy distribution for sampling actions and ensembles and computing probabilistic measures."""
    def __init__(self, action_spec, policy=None, ActionExtractor=None, discrete=False, temp_schedule=None,
                 optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.discrete = discrete
        self.action_spec = action_spec

        # Standard dev value
        self.temp_schedule = temp_schedule

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
    def Omega(self, mean, stddev, step=1):
        # Optionally create policy from recipe
        return Utils.instantiate(self.policy, mean=mean, stddev=stddev, step=step, action_spec=self.action_spec,
                                 discrete=self.discrete, temp_schedule=self.temp_schedule,
                                 ActionExtractor=self.ActionExtractor) or \
            MonteCarlo(self.action_spec, mean, stddev, step, self.discrete, self.temp_schedule, self.ActionExtractor)


# Policy
class MonteCarlo(nn.Module):
    """Exploration and exploitation policy distribution compatible with discrete and continuous spaces and ensembles."""
    def __init__(self, action_spec, mean, stddev, step=1, discrete=False, temp_schedule=1, ActionExtractor=None):
        super().__init__()

        self.discrete = discrete
        self.discrete_as_continuous = action_spec.discrete and not self.discrete

        self.low, self.high = action_spec.low, action_spec.high

        if self.discrete:
            self.All_Qs = mean  # [b, e, n, d]
        else:
            self.mean = mean  # [b, e, n, d]

        self.stddev = stddev

        # A secondary mapping that is applied after sampling an continuous-action
        self.ActionExtractor = ActionExtractor or nn.Identity()

        if self.discrete_as_continuous:
            self.temp = Utils.schedule(temp_schedule, step)  # Temp for controlling entropy of re-sample

    # SubPolicy
    @cached_property
    def Psi(self):
        if self.discrete:
            # Pessimistic Q-values per action
            logits, ind = self.All_Qs.min(1)  # Min-reduced ensemble [b, n, d]

            # Corresponding entropy temperature
            stddev = torch.tensor(self.stddev) if isinstance(self.stddev, float) \
                else Utils.gather(self.stddev, ind.unsqueeze(1), 1, 1).flatten(1).sum(1)  # Min-reduced ensemble std [b]

            try:
                return NormalizedCategorical(logits=logits, low=self.low, high=self.high, temp=stddev, dim=-2)
            except ValueError:
                # M1 Mac MPS can sometimes throw ValueError on log-sum-exp trick
                Psi = NormalizedCategorical(logits=logits.to('cpu'), low=self.low, high=self.high, temp=stddev, dim=-2)
                Psi.logits = Psi.logits.to(logits.device)
                return Psi
        else:
            return TruncatedNormal(self.mean, self.stddev, low=self.low, high=self.high, stddev_clip=0.3)

    def log_prob(self, action):
        # Log-probability
        log_prob = self.Psi.log_prob(action)  # (Log-space is more numerically stable)

        # If continuous-action is a discrete distribution, it gets double-sampled
        if self.discrete_as_continuous:
            log_prob += action / self.temp  # (Adding log-probs is equivalent to multiplying probs)

        return log_prob  # [b, e]

    @cached_property
    def _entropy(self):
        return self.Psi.entropy() if self.discrete else self.Psi.entropy().mean(-1)  # [b, e]

    def entropy(self, action=None):
        # If continuous-action is a discrete distribution, 2nd sample also has entropy
        if self.discrete_as_continuous:
            # Approximate joint entropy
            return self._entropy + torch.distributions.Categorical(logits=action / self.temp).entropy()

        return self._entropy  # [b, e]

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
            return action[:, torch.randint(action.shape[1], [])]  # Uniform sample again across ensemble

        # If sampled action is a discrete distribution, sample again
        if self.discrete_as_continuous:
            action = torch.distributions.Categorical(logits=action / self.temp).sample()  # Sample again

        return action

    def rsample(self, sample_shape=None):
        return self.sample(sample_shape, detach=False)  # Differentiable explore

    # Exploitation policy
    @cached_property
    def best(self):
        # Absolute Determinism (kind of)

        # Argmax for discrete, extract action for continuous  (Note: Ensemble reduce somewhat random; could use critic)
        action = self.Psi.normalize(self.Psi.logits.argmax(-1, keepdim=True).transpose(-1, -2)) if self.discrete \
            else self.ActionExtractor(self.mean[:, torch.randint(self.mean.shape[1], [])])  # Extract ensemble-reduce

        # If continuous-action is a discrete distribution
        if self.discrete_as_continuous:
            action = action.argmax(-1)  # Argmax

        return action
