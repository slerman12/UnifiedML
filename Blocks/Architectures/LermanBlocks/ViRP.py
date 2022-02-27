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


class ViRP(ViT):
    def __init__(self, input_shape, patch_size=4, out_channels=32, heads=8, depth=3, pool='cls', output_dim=None,
                 experiment='head_head_in_RN'):
        super().__init__(input_shape, patch_size, out_channels, heads, depth, pool, True, output_dim)

        if experiment == 'concat_plus_in':  # velocity reasoning from mlp only
            core = RelationConcat
        elif experiment == 'plus_in_concat_plus_mid':  # see if attention is useful as "reason-er"
            core = RelationConcatV2
        elif experiment == 'head_wise_ln':  # disentangled relational reasoning - are the heads independent?
            core = RelationDisentangled
        elif experiment == 'head_in_RN':  # invariant relational reasoning between input-head - are they?
            core = RelationSimpler
        elif experiment == 'head_head_in_RN':  # relational reasoning between heads
            core = RelationRelative
        elif experiment == 'head_head_RN_plus_in':  # does reason-er only need heads independent of input/tokens?
            core = RelationSimplerV3
        else:
            # layernorm values, confidence
            # see if more mhdpa layers picks up the load - is the model capacity equalized when layers are compounded?
            core = RelationRelative

        self.attn = nn.Sequential(*[core(out_channels, heads) for _ in range(depth)])


# Concat, then residual from input
class RelationConcat(nn.Module):
    def __init__(self, dim=32, heads=8, context_dim=None, value_dim=None):
        super().__init__()

        context_dim = dim if context_dim is None \
            else context_dim

        value_dim = dim if value_dim is None \
            else value_dim

        self.heads = math.gcd(8, value_dim) if heads is None \
            else heads

        self.context_dim = context_dim
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

        attn = self.LN_mid(self.attn(x, context))  # Relation
        out = self.LN_out(self.mlp(attn, x)) + x  # Reason-er

        return out


# Input to middle residual, concat, then middle to out residual
class RelationConcatV2(RelationConcat):
    def forward(self, x, context=None):
        if context is None:
            context = x

        attn = self.LN_mid(self.attn(x, context)) + x  # Attention = Reason-er
        out = self.LN_out(self.mlp(attn, x)) + attn  # MLP = Course Corrector

        return out


# Heads layer norm'd
class RelationDisentangled(RelationConcat):
    def __init__(self, dim=32, heads=8, context_dim=None, value_dim=None):
        super().__init__(dim, heads, context_dim, value_dim)

        self.LN_mid = nn.LayerNorm(self.value_dim // self.heads)

    def forward(self, x, context=None):
        if context is None:
            context = x

        attn = self.attn(x, context)
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)
        norm = self.LN_mid(head_wise)
        disentangled = norm.view(attn.shape)

        out = self.LN_out(self.mlp(disentangled, x)) + x

        return out


# Head-in
class RelationSimpler(RelationDisentangled):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None):
        super().__init__(dim, heads, context_dim, value_dim)

        self.RN = RN(dim, dim)

    def forward(self, x, context=None):
        if context is None:
            context = x  # [b, n, d]

        attn = self.attn(x, context)  # [b, n, h * d]
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)  # [b, n, h, d]

        norm = self.LN_mid(head_wise)  # [b, n, h, d]
        residual = x.unsqueeze(-2)  # [b, n, 1, d]

        relation = norm.flatten(0, -3)  # [b * n, h, d]
        residual = residual.flatten(0, -3)  # [b * n, 1, d]

        out = self.LN_out(self.RN(relation, residual))  # [b * n, d]

        return out.view(x.shape) + x  # [b, n, d]


# Head-head:in
class RelationRelative(RelationDisentangled):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None):
        super().__init__(dim, heads, context_dim, value_dim)

        self.attn = ReLA(dim, self.heads, self.context_dim, self.value_dim * self.heads)
        self.RN = RN(dim, dim * 2)

    def forward(self, x, context=None):
        if context is None:
            context = x  # [b, n, d]

        attn = self.attn(x, context)  # [b, n, h * d]
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)  # [b, n, h, d]
        print(head_wise.shape)

        norm = self.LN_mid(head_wise)  # [b, n, h, d]
        residual = x.unsqueeze(-2)  # [b, n, 1, d]

        relation = norm.flatten(0, -3)  # [b * n, h, d]
        residual = residual.flatten(0, -3)  # [b * n, 1, d]
        context = torch.cat([residual.expand(*relation.shape[:-1], -1), relation], -1)  # [b * n, h, d * 2]

        out = self.LN_out(self.RN(relation, context))  # [b * n, d]

        return out.view(x.shape) + x  # [b, n, d]

