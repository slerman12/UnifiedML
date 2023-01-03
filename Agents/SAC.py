# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

from Blocks.Architectures import MLP

import Utils

from Agents import DQNAgent


class DuelingDQNAgent(DQNAgent):
    """
    Dueling Deep Q Networks
    (https://arxiv.org/abs/1511.06581)
    """
    def __init__(self, hidden_dim, recipes, **kwargs):
        # Use dueling architecture
        recipes.actor.Pi_head = create(hidden_dim=hidden_dim, Pi_head=recipes.actor.Pi_head)

        super().__init__(hidden_dim=None, recipes=recipes, **kwargs)


def create(hidden_dim=1024, Pi_head=None):
    class DuelingDQN(torch.nn.Module):
        """Dueling Architecture"""
        def __init__(self, input_shape=50, output_shape=(2,)):
            super().__init__()

            self.V = Utils.instantiate(Pi_head, input_shape=input_shape,
                                       output_shape=output_shape) or MLP(input_shape, 1, hidden_dim, 2)  # Default, MLP

            self.A = Utils.instantiate(Pi_head, input_shape=input_shape,
                                       output_shape=output_shape) or MLP(input_shape, output_shape, hidden_dim, 2)

        def forward(self, obs):
            # Value, Advantage
            V, A = self.V(obs), self.A(obs)

            # Q-Value
            return V + (A - A.mean())

    return DuelingDQN
