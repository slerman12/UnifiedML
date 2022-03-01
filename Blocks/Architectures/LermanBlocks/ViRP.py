# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from einops import rearrange
from opt_einsum_torch import EinsumPlanner
from torch import nn

import Utils
from Blocks.Architectures import MLP
from Blocks.Architectures.MultiHeadAttention import ReLA, TokenAttentionBlock, AttentionPool
from Blocks.Architectures.Perceiver import TokenAttention
from Blocks.Architectures.RN import RN
from Blocks.Architectures.Vision.ViT import ViT


class ViRP(ViT):
    def __init__(self, input_shape, patch_size=4, out_channels=32, heads=8, depth=3, pool='cls', output_dim=None,
                 experiment='relation_block',
                 ViRP=True  # perceiver cross-attend, currently hurts
                 ):

        self.ViRP = ViRP
        if ViRP:
            self.tokens_per_axis = 50

        super().__init__(input_shape, patch_size, out_channels, heads, depth, pool, True, output_dim)

        if experiment == 'concat_plus_in':  # ! Velocity reasoning from mlp only
            core = RelationConcat
        elif experiment == 'plus_in_concat_plus_mid':  # See if attention is useful as "Reason-er"
            core = RelationConcatV2
        elif experiment == 'head_wise_ln':  # Disentangled relational reasoning - are the heads independent, equal vote?
            core = RelationDisentangled
        elif experiment == 'head_in_RN':  # Invariant relational reasoning between input-head - are they, period?
            core = RelationSimpler
        elif experiment == 'head_head_RN_plus_in':  # Does reason-er only need heads independent of input/tokens?
            core = RelationSimplerV2
        elif experiment == 'head_head_in_RN':  # ! Relational reasoning between heads
            core = RelationRelative
        elif experiment == 'head_head_in_RN_small':  # ! Relational reasoning between heads, smaller RN
            core = RelationRelativeV2
        elif experiment == 'relation_block':
            core = RelationBlock
        # else:
        #     # ! layernorm values, confidence
        #     # see if more mhdpa layers picks up the load - is the model capacity equalized when layers are compounded?
        #     core = RelationRelative

        self.attn = nn.Sequential(*[core(out_channels, heads) for _ in range(depth)])

        if ViRP:
            tokens = self.tokens_per_axis ** 2
            self.attn = nn.Sequential(TokenAttentionBlock(out_channels, heads, tokens, relu=True),
                                      *[core(out_channels, heads) for _ in range(depth)])

    def repr_shape(self, c, h, w):
        if self.ViRP:
            return self.out_channels, self.tokens_per_axis, self.tokens_per_axis
        return super().repr_shape(c, h, w)


# Vision Perceiver
class ViPer(ViT):
    def __init__(self, input_shape, patch_size=4, out_channels=32, heads=8, depth=3, pool='cls', output_dim=None):
        super().__init__(input_shape, patch_size, out_channels, heads, depth, pool, True, output_dim)

        self.attn = nn.Sequential(TokenAttentionBlock(out_channels, heads, 100, relu=True),
                                  *[RelationRelativeV2(out_channels, heads) for _ in range(depth)])


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
class RelationSimpler(RelationConcat):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None):
        super().__init__(dim, heads, context_dim, value_dim)

        self.attn = ReLA(dim, self.heads, self.context_dim, self.value_dim * self.heads)
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
class RelationRelative(RelationConcat):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None):
        super().__init__(dim, heads, context_dim, value_dim)

        self.attn = ReLA(dim, self.heads, self.context_dim, self.value_dim * self.heads)
        self.RN = RN(dim, dim * 2)

    def forward(self, x, context=None):
        if context is None:
            context = x  # [b, n, d]

        attn = self.attn(x, context)  # [b, n, h * d]
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)  # [b, n, h, d]

        norm = self.LN_mid(head_wise)  # [b, n, h, d]
        residual = x.unsqueeze(-2)  # [b, n, 1, d]

        relation = norm.flatten(0, -3)  # [b * n, h, d]
        residual = residual.flatten(0, -3)  # [b * n, 1, d]
        context = torch.cat([residual.expand(*relation.shape[:-1], -1), relation], -1)  # [b * n, h, d * 2]

        out = self.LN_out(self.RN(relation, context))  # [b * n, d]

        return out.view(x.shape) + x  # [b, n, d]# Head-head:in


# Smaller RN
class RelationRelativeV2(RelationRelative):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None):
        super().__init__(dim, heads, context_dim, value_dim)

        self.RN = RN(dim, dim * 2, inner_depth=0, outer_depth=0, mid_nonlinearity=nn.ReLU(inplace=True))


# Head-head:in from tokens
class RelationBlock(RelationRelativeV2):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None):
        super().__init__(dim, heads, context_dim, value_dim)

        self.attn = Relation(dim, self.heads, self.context_dim, self.value_dim * self.heads)


# Pools features relationally, in linear time
class RelationPool(nn.Module):
    def __init__(self, channels_in=32, output_dim=None, input_shape=None):
        super().__init__()

        self.input_shape = input_shape

        if input_shape is not None:
            channels_in = input_shape[-3]

        if output_dim is None:
            output_dim = channels_in

        self.pool = nn.Sequential(Utils.ChSwap,
                                  TokenAttention(channels_in, tokens=16, value_dim=output_dim, relu=True),
                                  nn.LayerNorm(output_dim),
                                  RN(output_dim))

    def forward(self, x):
        return self.pool(x)


class Relation(nn.Module):
    def __init__(self, dim=32, heads=None, context_dim=None, value_dim=None, talk_h=False, relu=False):
        super().__init__()

        self.dim = dim

        context_dim = dim if context_dim is None \
            else context_dim

        value_dim = dim if value_dim is None \
            else value_dim

        heads = math.gcd(8, value_dim) if heads is None \
            else heads

        self.value_dim = value_dim
        self.heads = heads

        assert value_dim % heads == 0, f'value dim={dim} is not divisible by heads={heads}'

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim + value_dim, bias=False)

        self.relu = nn.ReLU(inplace=True) if relu else None

        # "Talking heads" (https://arxiv.org/abs/2003.02436)
        self.talk_h = nn.Sequential(Utils.ChSwap, nn.Linear(heads, heads, bias=False),
                                    nn.LayerNorm(heads), Utils.ChSwap) if talk_h else nn.Identity()

    def forward(self, x, context=None):
        # Conserves shape
        shape = x.shape
        assert shape[-1] == self.dim, f'input dim ≠ pre-specified {shape[-1]}≠{self.dim}'

        if context is None:
            context = x

        tokens = len(x.shape) == 2
        if not tokens:
            x = x.flatten(1, -2)
        context = context.flatten(1, -2)

        q = self.to_q(x)
        k, v = self.to_kv(context).tensor_split([self.dim], dim=-1)

        # Note: I think it would be enough for the key to have just a single head
        pattern = 'n (h d) -> h n d' if tokens else 'b n (h d) -> b h n d'
        q = rearrange(q, pattern, h=self.heads)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (k, v))

        # Memory efficient toggle, e.g., =0.5
        mem_limit = False
        einsum = EinsumPlanner(q.device, cuda_mem_limit=mem_limit).einsum if 0 < mem_limit < 1 \
            else torch.einsum

        pattern = 'h i d, b h j d -> b h i j' if tokens else 'b h i d, b h j d -> b h i j'
        self.dots = einsum(pattern, q, k) * self.dim ** -0.5

        weights = self.dots.softmax(dim=-1) if self.relu is None else self.relu(self.dots)

        # "Talking heads"
        weights = self.talk_h(weights)

        attn = einsum('b h i j, b h j d -> b h i d', weights, v)
        rtn = torch.argmax(weights, dim=-1)  # [b, h, i]
        rtn = Utils.gather_indices(v, rtn, dim=-2)  # [b, h, i, d]
        rtn = attn - (attn - rtn).detach()

        out = rearrange(rtn, 'b h n d -> b n (h d)')

        # Restores original shape
        if not tokens:
            out = out.view(*shape[:-1], -1)

        if 0 < mem_limit < 1:
            out = out.to(q.device)

        return out

