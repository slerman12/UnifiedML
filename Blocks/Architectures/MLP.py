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
            else math.prod(input_shape) if flatten \
            else input_shape[-1]

        self.output_dim = output_dim

        self.MLP = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(self.input_dim if i == 0 else hidden_dim,
                          hidden_dim if i < depth else output_dim),
                non_linearity if i < depth else nn.Sigmoid() if binary else nn.Identity(),
                nn.Dropout(dropout) if i < depth else nn.Identity())
            for i in range(depth + 1)])

        self.flatten = flatten  # Proprioceptive

        self.apply(Utils.weight_init)

    def repr_shape(self, *_):
        if self.flatten:
            return self.output_dim, 1, 1  # Dummy 1s
        return *_[:-1], self.output_dim

    def forward(self, *x):
        # Computes batch dims
        batch_dims = (1,)
        for obs in x:
            if len(obs.shape) > 1:
                batch_dims = obs.shape[0:1 if self.flatten else -1]
                break

        # Give each obs a uniform batch dim, flatten, and concatenate
        # If flatten is False, will operate on last axis only
        obs = torch.cat([(obs.expand(*batch_dims, *obs.shape) if len(obs.shape) < len(batch_dims) + 1
                          else obs).flatten(1 if self.flatten else -1) for obs in x], -1)

        assert obs.shape[-1] == self.input_dim, f'MLP input dim {self.input_dim} does not match provided ' \
                                                f'input dim {obs.shape[-1]} of observation(s) ' \
                                                f'with shape(s): {" ".join([obs.shape for obs in x])}'

        return self.MLP(obs)
