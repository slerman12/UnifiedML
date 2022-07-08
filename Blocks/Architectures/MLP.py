# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

import Utils


class MLP(nn.Module):
    def __init__(self, input_shape=(128,), output_dim=1024, hidden_dim=512, depth=1, non_linearity=nn.ReLU(True),
                 dropout=0, binary=False, flatten=False):
        super().__init__()

        self.input_dim = input_shape if isinstance(input_shape, int) \
            else math.prod(input_shape)  # Assumes input flattened, 1D

        self.output_dim = output_dim

        self.MLP = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(self.input_dim if i == 0 else hidden_dim,
                          hidden_dim if i < depth else output_dim),
                non_linearity if i < depth else nn.Sigmoid() if binary else nn.Identity(),
                nn.Dropout(dropout) if i < depth else nn.Identity())
            for i in range(depth + 1)])

        self.flatten = flatten  # Flattens (should be set to True if input not flattened to 1D but proprioceptive)

        self.apply(Utils.weight_init)

    def repr_shape(self, *_):
        if self.flatten:
            return self.output_dim, 1, 1  # Dummy 1s
        return *_[:-1], self.output_dim

    def forward(self, obs, *context):
        # Assumes context can be concatenated along last dim
        return self.MLP(torch.cat([obs, *context], -1).flatten(1 if self.flatten else -1))
