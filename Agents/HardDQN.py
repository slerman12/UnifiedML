# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

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
    A policy where returned probabilities correspond to hard exploitation policy rather than exploration for Q-learning
    Can use with AC2Agent: python Run.py Policy=Agents.HardDQN.HardDQNPolicy
    """
    # Log-probability to multiply each action's Q-value by to estimate expected future Q-value
    def log_prob(self, action=None):
        if self.discrete:
            num_actions = self.Psi.logits.size(-1)
            return one_hot(self.best.sum(-1), num_actions, null_value=-math.inf)  # Learn exploitative Q-value-target

        return super().log_prob(action)  # For compatibility with continuous spaces/Agents
