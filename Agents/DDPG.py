# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

import Utils

from Agents import DrQV2Agent


class DDPGAgent(DrQV2Agent):
    """
    Deep Deterministic Policy Gradient
    (https://arxiv.org/pdf/1509.02971.pdf)
    """

    def __init__(self, recipes, stddev_schedule, **kwargs):
        recipes.aug = torch.nn.Identity()

        super().__init__(recipes=recipes, stddev_schedule=None, **kwargs)  # Use specified start sched


class OUNoiseSchedule(torch.nn.Module):
    """
    Ornstein-Uhlenbeck Process: https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        super().__init__()

        self.x = None
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.x = torch.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.x
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(self.action_dim)
        self.x = x + dx
        return self.x

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return torch.clip(action + ou_state, self.low, self.high)
