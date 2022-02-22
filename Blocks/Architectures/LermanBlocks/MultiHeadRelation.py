# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

import Utils

from Blocks.Architectures import MLP
from Blocks.Architectures.MultiHeadAttention import ReLA, AttentionPool


class Relate(nn.Module):
    """Relation Is All You Need
        - MHDPR: Multi-Head Dot-Product Relation"""
    def __init__(self, dim=32, guides=32, guide_dim=None, heads=1, input_shape=None):
        super().__init__()

        if input_shape is not None:
            dim = input_shape[-3]

        guide_dim = dim if guide_dim is None \
            else guide_dim

        self.guides = nn.Parameter(torch.randn((guides, dim)))
        # ReLA: Rectified linear attention, https://arxiv.org/pdf/2104.07012.pdf
        self.ReLA = ReLA(guide_dim, heads, dim, dim * heads)
        self.LN = nn.LayerNorm(dim)
        self.relate = RN(dim)

        nn.init.kaiming_normal_(self.guides)

    def repr_shape(self, c, h, w):
        return c, 1, 1

    def forward(self, x):
        guides = self.guides.expand(x.shape[0], -1, -1)
        attn = self.ReLA(guides, x)
        head_wise = attn.view(*attn.shape[:-2], -1, x.shape[-1])
        norm = self.LN(head_wise)  # Maybe add the top-1 of x as a residual

        out = self.relate(norm)

        return out


class RelationPool(AttentionPool):
    """CNN feature pooling with MHDPR"""
    def __init__(self, channels_in=32, heads=None, output_dim=None, guides=32, input_shape=None):
        super().__init__()

        if input_shape is not None:
            channels_in = input_shape[-3]

        if heads is None:
            heads = math.gcd(output_dim, 8)  # Approx 8

        self.pool = nn.Sequential(Utils.ChSwap,
                                  Relate(channels_in, guides, output_dim, heads),
                                  Utils.ChSwap)


class RN(nn.Module):
    """Relation Network https://arxiv.org/abs/1706.01427"""
    def __init__(self, dim, inner_depth=3, outer_depth=2, hidden_dim=None, output_dim=None, input_shape=None):
        super().__init__()

        if input_shape is not None:
            dim = input_shape[-3]

        if hidden_dim is None:
            hidden_dim = 2 * dim

        self.output_dim = dim if output_dim is None \
            else output_dim

        self.inner = MLP(dim, hidden_dim, hidden_dim, inner_depth)
        self.outer = MLP(hidden_dim, self.output_dim, hidden_dim, outer_depth) \
            if outer_depth else None

    def repr_shape(self, c, h, w):
        return self.output_dim, 1, 1

    def forward(self, x):
        x = x.flatten(1, -2)

        pair_a = x.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
        pair_b = x.unsqueeze(2).expand(-1, -1, x.shape[1], -1)
        pair = torch.cat([pair_a, pair_b], -1)

        relations = self.inner(pair)

        out = relations if self.outer is None \
            else self.outer(relations.sum(1).sum(1))

        return out


"""Experimental"""
# class SelfRelate(nn.Module):  # Do it Perceiver style: Cross attend, then self attend, double block
#     """Relation Is All You Need
#         - MHDPR: Multi-Head Dot-Product Relation"""
#     def __init__(self, dim=32, heads=1, input_shape=None, output_dim=None):
#         super().__init__()
#
#         if input_shape is not None:
#             dim = input_shape[-3]
#
#         self.output_dim = output_dim
#
#         # ReLA: Rectified linear attention, https://arxiv.org/pdf/2104.07012.pdf
#         self.ReLA = CrossAttention(dim, heads, dim, dim * heads, relu=True)
#         self.LN = nn.LayerNorm(dim)
#         self.relate = RN(dim, outer_depth=0 if output_dim is None else 3)
#
#         if output_dim is None:
#             self.LN_out = nn.LayerNorm(dim)
#
#         nn.init.kaiming_normal_(self.guides)
#
#     def repr_shape(self, c, h, w):
#         return c, 1, 1
#
#     def forward(self, x):
#         attn = self.ReLA(x, x)
#         head_wise = attn.view(*attn.shape[:-2], -1, x.shape[-1])
#         norm = self.LN(head_wise)
#         residuals = norm + x.unsqueeze(-2)
#
#         out = self.relate(residuals)  # Maybe suffices just to do residuals head-wise, otherwise standard attention
#
#         if self.output_dim is None:
#             out = self.LN_out((self.ReLA.dots.unsqueeze(-1) * out).mean(-3)) + residuals
#
#         return out
