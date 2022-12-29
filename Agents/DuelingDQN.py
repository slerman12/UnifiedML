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
    def __init__(self, action_spec, hidden_dim, trunk_dim, recipes, **kwargs):
        # Compatible with continuous and multi-action discrete envs
        out_shape = [(action_spec.discrete_bins or 1) * action_spec.shape[0], *action_spec.shape[1:]]

        # Use dueling architecture
        recipes.actor.Pi_head = DuelingDQN(trunk_dim, out_shape, hidden_dim=hidden_dim,
                                           ensemble_size=2, Pi_head=recipes.actor.Pi_head)

        super().__init__(action_spec=action_spec, hidden_dim=hidden_dim, trunk_dim=trunk_dim, recipes=recipes, **kwargs)


class DuelingDQN(torch.nn.Module):
    """Dueling Architecture"""
    def __init__(self, input_shape=50, output_shape=(2,), hidden_dim=1024, ensemble_size=1, Pi_head=None):
        super().__init__()

        self.V = Utils.Ensemble([Utils.instantiate(Pi_head, i, input_shape=input_shape, output_shape=output_shape) or
                                 MLP(input_shape, 1, hidden_dim, 2) for i in range(ensemble_size)])

        self.A = Utils.Ensemble([Utils.instantiate(Pi_head, i, input_shape=input_shape, output_shape=output_shape) or
                                 MLP(input_shape, output_shape, hidden_dim, 2) for i in range(ensemble_size)])

    def forward(self, obs):
        # Value, Advantage
        V, A = self.V(obs), self.A(obs)

        # Q-Value
        return V + (A - A.mean())
