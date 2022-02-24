# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

from torch import nn

from Blocks.Architectures import MLP
from Blocks.Architectures.MultiHeadAttention import ReLA
from Blocks.Architectures.Vision.ViT import ViT


class RelationSimplest(nn.Module):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None):
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

        self.ln_mid = nn.LayerNorm(value_dim)
        self.ln_out = nn.LayerNorm(value_dim)

    def repr_shape(self, c, h, w):
        return self.value_dim, h, w  # Assumes channels last

    def forward(self, x, context=None):
        if context is None:
            context = x

        attn = self.ln_mid(self.attn(x, context))
        out = self.ln_out(self.mlp(attn, x)) + attn

        return out


class ViRPSimplest(ViT):
    def __init__(self, input_shape, patch_size=4, out_channels=32, heads=8, depth=3, pool='cls', output_dim=None):
        super().__init__(input_shape, patch_size, out_channels, heads, depth, pool, output_dim)

        self.attn = nn.Sequential(*[RelationSimplest(out_channels, heads) for _ in range(depth)])


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
