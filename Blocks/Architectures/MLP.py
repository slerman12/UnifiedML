# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

import Utils


class MLP(nn.Module):
    """
    MLP Architecture generalized to broadcast input shapes
    """
    def __init__(self, input_shape=(128,), output_dim=1024, hidden_dim=512, depth=1, non_linearity=nn.ReLU(True),
                 dropout=0, binary=False):
        super().__init__()

        self.input_shape = (input_shape,) if isinstance(input_shape, int) \
            else input_shape
        self.input_dim = math.prod(self.input_shape)  # If not already flattened/1D, will try to auto-flatten

        self.output_dim = output_dim

        self.MLP = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(self.input_dim if i == 0 else hidden_dim,
                          hidden_dim if i < depth else output_dim),  # Linear
                non_linearity if i < depth else nn.Sigmoid() if binary else nn.Identity(),  # Activation
                nn.Dropout(dropout) if i < depth else nn.Identity())  # Dropout
            for i in range(depth + 1)])

        # Initialize weights
        self.apply(Utils.weight_init)

    def repr_shape(self, c, w, h):
        flatten = -1 if h == self.input_dim \
            else -len(self.input_shape)  # Auto-flatten if needed

        return *[size for size in (c, w, h) if size is not None][:flatten], self.output_dim

    def forward(self, *obs):
        obs = torch.cat(obs, -1)  # Assumes inputs can be concatenated along last dim

        flatten = -1 if obs.shape[-1] == self.input_dim \
            else -len(self.input_shape)

        return self.MLP(obs.flatten(flatten))  # Auto-flatten if needed
