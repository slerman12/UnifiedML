# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch import nn

from Blocks.Architectures import MLP
from Blocks.Architectures.MultiHeadAttention import CrossAttend


class Relate(nn.Module):
    """Relation Is All You Need"""
    def __init__(self, dim=32, guides=4, guide_dim=None, heads=1, input_shape=None):
        super().__init__()

        if input_shape is not None:
            dim = input_shape[-3]

        guide_dim = dim if guide_dim is None \
            else guide_dim

        self.guides = nn.Parameter(torch.randn((guides, dim)))
        self.attn = CrossAttend(guide_dim, heads, dim, dim * heads)
        self.rltn = RelationPool(dim)

        nn.init.kaiming_normal_(self.guides)

    def repr_shape(self, c, h, w):
        return c, 1, 1

    def forward(self, x):
        guides = self.guides.expand(x.shape[0], -1, -1)
        attn = self.attn(guides, x)
        head_wise = attn.view(*attn.shape[:-2], -1, x.shape[-1])

        out = self.rltn(head_wise)

        return out


class RelationPool(nn.Module):
    def __init__(self, dim, inner_depth=3, outer_depth=2, hidden_dim=None, output_dim=None, input_shape=None):
        super().__init__()

        if input_shape is not None:
            dim = input_shape[-3]

        if hidden_dim is None:
            hidden_dim = 2 * dim

        self.output_dim = dim if output_dim is None \
            else output_dim

        self.inner = MLP(dim, hidden_dim, hidden_dim, inner_depth)
        self.outer = MLP(hidden_dim, self.output_dim, hidden_dim, outer_depth)

    def repr_shape(self, c, h, w):
        return self.output_dim, 1, 1

    def forward(self, x):
        x = x.flatten(1, -2)

        pair_a = x.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        pair_b = x.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        pair = torch.cat([pair_a, pair_b], -1)

        relations = self.inner(pair)

        out = self.outer(relations.sum(1).sum(1))

        return out
