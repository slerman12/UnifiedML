# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

from Blocks.Architectures import MLP
from Blocks.Architectures.MultiHeadAttention import CrossAttend


class RelationPool(nn.Module):
    def __init__(self, dim, axes=0, inner_depth=3, outer_depth=2, hidden_dim=None, output_dim=None, input_shape=None):
        super().__init__()

        if input_shape is not None:
            dim = input_shape[-3]

        self.axes = axes

        if hidden_dim is None:
            hidden_dim = 2 * dim

        self.output_dim = dim if output_dim is None \
            else output_dim

        self.inner = MLP(dim + axes, hidden_dim, hidden_dim, inner_depth)
        self.outer = MLP(hidden_dim, self.output_dim, hidden_dim, outer_depth)

    def repr_shape(self, c, h, w):
        return self.output_dim, 1, 1

    def forward(self, x, locality=None, localize=False):
        # mid_shape = x.shape[1:-1]  # Assumes channels-last
        #
        # if localize:
        #     assert len(mid_shape) == self.axes
        #
        #     if locality is None:
        #         axis_localities = []
        #         for dim in mid_shape:
        #             axis_localities.append(torch.linspace(-dim // 2, dim - dim // 2, dim))
        #         locality = torch.meshgrid(axis_localities).unsqueeze(-1).expand_as(x)
        #     x = torch.cat([x, locality], -1)

        x = x.flatten(1, -2)

        pair_a = x.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        pair_b = x.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        pair = torch.cat([pair_a, pair_b], -1)

        relations = self.inner(pair)

        out = self.outer(relations.sum(1).sum(1))

        return out


class RelationAttend(nn.Module):
    def __init__(self, dim=32, heads=1, axes=0, context_dim=None, input_shape=None):
        super().__init__()

        if input_shape is not None:
            dim = input_shape[-3]

        context_dim = dim if context_dim is None \
            else context_dim

        self.attn = CrossAttend(dim, heads, context_dim, dim * heads)
        self.rltn = RelationPool(dim, axes)

    def repr_shape(self, c, h, w):
        return self.value_dim, h, w  # Assumes channels last

    def forward(self, x, context=None):
        if context is None:
            context = x

        attn = self.attn(x, context)
        head_wise = attn.view(*attn.shape[:-2], -1, x.shape[-1])

        out = self.rltn(head_wise)

        return out
