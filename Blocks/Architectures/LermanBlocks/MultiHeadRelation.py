# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

from Blocks.Architectures import MLP
from Blocks.Architectures.MultiHeadAttention import ReLA
from Blocks.Architectures.RN import RN
from Blocks.Architectures.Vision.ViT import ViT


# Concatenate, add middle to last
class RelationSimplest(nn.Module):
    def __init__(self, dim=32, heads=8, context_dim=None, value_dim=None):
        super().__init__()

        context_dim = dim if context_dim is None \
            else context_dim

        value_dim = dim if value_dim is None \
            else value_dim

        self.heads = math.gcd(8, value_dim) if heads is None \
            else heads

        self.value_dim = value_dim

        self.attn = ReLA(dim, self.heads, context_dim, value_dim)
        self.mlp = MLP(value_dim + dim, value_dim, value_dim, 1, nn.GELU())

        self.LN_mid = nn.LayerNorm(value_dim)
        self.LN_out = nn.LayerNorm(value_dim)

    def repr_shape(self, c, h, w):
        return self.value_dim, h, w  # Assumes channels last

    def forward(self, x, context=None):
        if context is None:
            context = x

        attn = self.LN_mid(self.attn(x, context))
        # attn = self.LN_mid(self.attn(x, context)) + x  # TODO Test
        out = self.LN_out(self.mlp(attn, x)) + attn

        return out


# Concat, then residual from input
class RelationSimplestV2(RelationSimplest):
    def forward(self, x, context=None):
        if context is None:
            context = x

        attn = self.LN_mid(self.attn(x, context))
        out = self.LN_out(self.mlp(attn, x)) + x  # Result: this residual is better as x
        # Therefore, the model prefers to reason about velocity from x via the MLP, NOT via the attention
        # We conclude that the attention is rather used for facilitating the MLP's reasoning, which updates x
        # Therefore disentangling x from the attention is necessary
        # And furthermore, making the attention as easy to reason about as possible is important
        # To do the former, we concatenate instead of add -- since adding forces the attention to be a velocity, not
        # a representation that can be reasoned about independently
        # To do the latter, we apply the layer-norm head-wise, so as not to entangle the different heads,
        # And then relate the different heads invariably via a relation network consisting of an inner and outer
        # MLP, not just a single MLP. The former MLP is non-local. This way, more signal from the original
        # context is preserved and its original non-locality.

        # The residual is just x, the input.
        # We introduce a new term, velocity, to describe the counterpart to the residual.
        # How to change x -- the velocity on x -- is what we refer to by reasoning.

        # Conclusion: the MLP is better for reasoning <about the input!> than attention, but attention
        # serves to give the MLP what to reason about

        return out


# Input to middle residual, concat, then middle to out residual
class RelationSimplestV3(RelationSimplest):
    def forward(self, x, context=None):
        if context is None:
            context = x

        attn = self.LN_mid(self.attn(x, context)) + x  # TODO Test
        out = self.LN_out(self.mlp(attn, x)) + attn

        return out

# Two results:
# 1. Concat outperforms add, AND concat outperforms no residual
#     Conclusion: it helps to reason about x and the attention in relation to one another or independently, unaggregated
#         Also: the MLP is better for reasoning than attention, but attention serves to give the MLP
# #         # what to reason about
# 2. Residual from x outperforms residual from attention -haven't tested
#     Conclusion: the MLP is better for reasoning than attention, but attention serves to give the MLP
#         # what to reason about - not to do reasoning itself


# Heads layer norm'd
class RelationDisentangled(RelationSimplestV2):
    def __init__(self, dim=32, heads=8, context_dim=None, value_dim=None):
        super().__init__()

        context_dim = dim if context_dim is None \
            else context_dim

        value_dim = dim if value_dim is None \
            else value_dim

        self.heads = math.gcd(8, value_dim) if heads is None \
            else heads

        self.value_dim = value_dim

        self.attn = ReLA(dim, self.heads, context_dim, value_dim)
        self.mlp = MLP(value_dim + dim, value_dim, value_dim, 1, nn.GELU())

        self.LN_mid = nn.LayerNorm(value_dim / heads)
        self.LN_out = nn.LayerNorm(value_dim)

    def forward(self, x, context=None):
        if context is None:
            context = x

        attn = self.attn(x, context)
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)
        norm = self.LN_mid(head_wise)
        disentangled = norm.view(attn.shape)

        out = self.LN_out(self.mlp(disentangled, x)) + x

        return out


class RelationSimpler(RelationDisentangled):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None):
        super().__init__(dim, heads, context_dim, dim * heads)

        self.RN = RN(dim, dim)

    def forward(self, x, context=None):
        if context is None:
            context = x  # [b, n, d]

        attn = self.attn(x, context)  # [b, n, h * d]
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)  # [b, n, h, d]

        norm = self.LN_mid(head_wise)  # [b, n, h, d]
        residual = x.unsqueeze(-2)  # [b, n, 1, d]

        relation = norm.flatten(0, -2)  # [b * n, h, d]
        residual = residual.flatten(0, -2)  # [b * n, 1, d]

        out = self.LN_out(self.RN(relation, residual))  # [b * n, d]

        return out.view(x.shape) + x  # [b, n, d]


class RelationRelative(RelationDisentangled):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None):
        super().__init__(dim, heads, context_dim, dim * heads)

        self.RN = RN(dim, dim * 2)

    def forward(self, x, context=None):
        if context is None:
            context = x  # [b, n, d]

        attn = self.attn(x, context)  # [b, n, h * d]
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)  # [b, n, h, d]

        norm = self.LN_mid(head_wise)  # [b, n, h, d]
        residual = x.unsqueeze(-2)  # [b, n, 1, d]

        relation = norm.flatten(0, -2)  # [b * n, h, d]
        residual = residual.flatten(0, -2)  # [b * n, 1, d]
        context = torch.cat([residual.expand_as(relation), residual], -1)  # [b * n, h, d * 2]

        out = self.LN_out(self.RN(relation, context))  # [b * n, d]

        return out.view(x.shape) + x  # [b, n, d]


# Can also test no ViRP, just ViT without mid-residual, to exclude that as the cause
# No, the issue isn't the presence of x, but rather the ability of the model to reason about both x and the attention
# Counterfactual: the presence of x disturbs performance
# False
# Conclusion: the capacity of the model to reason about x and the attention in relation is the causal factor


# TODO: Perceiver-style: from token/s linear attention: ViRP-relational
#  Last: if time permits, reparameterization for both ViRP and ViRP-relative


# class RelativeAttention(nn.Module):
#     """Relation Is All You Need
#         - MHDPRA: Multi-Head Dot-Product Relative Attention"""
#     def __init__(self, dim=32, heads=1, context_dim=None, input_shape=None):
#         super().__init__()
#
#         if input_shape is not None:
#             dim = input_shape[-3]
#
#         context_dim = dim if context_dim is None \
#             else context_dim
#
#         # ReLA: Rectified linear attention, https://arxiv.org/pdf/2104.07012.pdf
#         self.ReLA = ReLA(dim, heads, context_dim, dim * heads)
#         self.LN_mid = nn.LayerNorm(dim)
#         self.relate = RN(dim)
#         self.LN_out = nn.LayerNorm(dim)
#
#     def repr_shape(self, c, h, w):
#         return c, h, w
#
#     def forward(self, x, context=None):
#         if context is None:
#             context = x  # [b, n, d]
#
#         attn = self.ReLA(x, context)  # [b, n, h * d]
#         head_wise = attn.view(*attn.shape[:-1], -1, x.shape[-1])  # [b, n, h, d]
#
#         norm = self.LN_mid(head_wise)  # [b, n, h, d]
#         residual = x.unsqueeze(-2).expand_as(head_wise)  # [b, n, h, d]
#
#         relation = norm.flatten(0, -2)  # [b * n, h, d]
#         residual = residual.flatten(0, -2)  # [b * n, h, d]
#
#         out = self.LN_out(self.pool(relation, residual))  # [b * n, d]
#
#         return out.view(x.shape) + skip  # [b, n, d]
#
#         # For example:
#         residual = norm.sum(-2) + x
#         return self.LN_out(self.mlp(residual)) + residual
#
#
# class Relate(nn.Module):
#     """Relation Is All You Need
#         - MHDPR: Multi-Head Dot-Product Relation"""
#     def __init__(self, dim=32, guides=32, guide_dim=None, heads=1, input_shape=None):
#         super().__init__()
#
#         if input_shape is not None:
#             dim = input_shape[-3]
#
#         guide_dim = dim if guide_dim is None \
#             else guide_dim
#
#         self.guides = nn.Parameter(torch.randn((guides, dim)))
#         # ReLA: Rectified linear attention, https://arxiv.org/pdf/2104.07012.pdf
#         self.ReLA = ReLA(guide_dim, heads, dim, dim * heads)
#         self.LN = nn.LayerNorm(dim)
#         self.relation = RN(dim)
#
#         nn.init.kaiming_normal_(self.guides)
#
#     def repr_shape(self, c, h, w):
#         return c, h, w
#
#     def forward(self, x):
#         guides = self.guides.expand(x.shape[0], -1, -1)  # Not compatible with guides
#         attn = self.ReLA(guides, x)
#         head_wise = attn.view(*attn.shape[:-2], -1, x.shape[-1])
#         norm = self.LN(head_wise)
#
#         residual = norm + x.unsqueeze(-1).expand_as(head_wise)
#
#         out = self.relation(residual.flatten(0, -2)) + residual
#         return out.view(x.shape)
#
#
# class RelateSimple(nn.Module):
#     """Relation Is All You Need
#         - MHDPR: Multi-Head Dot-Product Relation"""
#     def __init__(self, dim=32, guides=32, guide_dim=None, heads=1, input_shape=None):
#         super().__init__()
#
#         if input_shape is not None:
#             dim = input_shape[-3]
#
#         guide_dim = dim if guide_dim is None \
#             else guide_dim
#
#         self.guides = nn.Parameter(torch.randn((guides, dim)))
#         # ReLA: Rectified linear attention, https://arxiv.org/pdf/2104.07012.pdf
#         self.ReLA = ReLA(guide_dim, heads, dim, dim * heads)
#         self.LN = nn.LayerNorm(dim)
#         self.relate = RN(dim)
#         # self.relate = MLP(dim * heads, dim, dim)
#
#         self.LN_out = nn.LayerNorm(dim)
#
#         nn.init.kaiming_normal_(self.guides)
#
#     def repr_shape(self, c, h, w):
#         return c, 1, 1
#
#     def forward(self, x, guides=None):
#         if guides is None:
#             guides = self.guides.expand(x.shape[0], -1, -1)
#
#         attn = self.ReLA(guides, x)
#         head_wise = attn.view(*attn.shape[:-2], -1, x.shape[-1])  # Expensive - do per value, but compatible with guides
#         norm = self.LN(head_wise)  # (Maybe add the top-1 of x as a residual, or add residual from guides?)
#
#         per_value = norm = norm.flatten(0, -2)  # Like this
#         # per_value = norm = norm.flatten(-2)  # For MLP
#
#         out = self.relate(norm)  # Can simplify by just flattening and using ML
#
#         # return out
#
#         # Update the guides e.g.
#         return out, self.guides + self.LN_out(out.mean(0))
#
#
#
# class RelationPool(AttentionPool):
#     """CNN feature pooling with MHDPR"""
#     def __init__(self, channels_in=32, heads=None, output_dim=None, guides=32, input_shape=None):
#         super().__init__()
#
#         if input_shape is not None:
#             channels_in = input_shape[-3]
#
#         if heads is None:
#             heads = math.gcd(output_dim, 8)  # Approx 8
#
#         self.pool = nn.Sequential(Utils.ChSwap,
#                                   Relate(channels_in, guides, output_dim, heads),
#                                   Utils.ChSwap,
#                                   nn.AdaptiveAvgPool2d((1, 1)),
#                                   nn.Flatten(-3))


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
