# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from math import inf

from Blocks.Creator import MonteCarlo

from Utils import one_hot

from Agents import DQNAgent


class HardDQNAgent(DQNAgent):
    """Hard-Deep Q-Learning as in the original Nature paper"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.actor.creator.policy = HardDQNPolicy


class HardDQNPolicy(MonteCarlo):
    """
    A policy where returned probabilities for Q-learning correspond to a hard exploitation policy
    Can use with AC2Agent: python Run.py Policy=Agents.HardDQN.HardDQNPolicy
    """
    # Log-probability to multiply each action's Q-value by to estimate expected future Q-value
    def log_prob(self, action=None):
        if action is None:
            num_actions = self.All_Qs.size(-2)
            log_prob = one_hot(self.best.squeeze(-1), num_actions, null_value=-inf)  # Learn exploitative Q-value-target

            return log_prob  # One-hot prob

        return super().log_prob(action)  # For compatibility with continuous spaces/Agents
