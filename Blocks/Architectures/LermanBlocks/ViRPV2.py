# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

from einops import rearrange
from opt_einsum_torch import EinsumPlanner

import torch
from torch import nn

import Utils

from Blocks.Architectures import MLP, ViT
from Blocks.Architectures.MultiHeadAttention import ReLA
from Blocks.Architectures.RN import RN
from Blocks.Architectures.Perceiver import Perceiver


class ViRP(ViT):
    def __init__(self, input_shape, patch_size=4, out_channels=32, heads=8, tokens=100,
                 token_dim=32, depth=3, pool='cls', output_dim=None, experiment='relation', ViRS=False):
        self.tokens = tokens
        self.ViRS = ViRS

        super().__init__(input_shape, patch_size, out_channels, heads, depth, pool, True, output_dim)

        if experiment == 'relation':
            block = RelationBlock
        elif experiment == 'relative':
            block = RelativeBlock
        elif experiment == 'independent':
            block = IndependentHeadsBlock
        elif experiment == 'disentangled':
            block = Disentangled
        elif experiment == 'course_corrector':
            block = CourseCorrectorBlock
        elif experiment == 'concat':
            block = ConcatBlock
        else:
            raise NotImplementedError('No such experiment')

        self.P = Perceiver(out_channels, heads, tokens, token_dim, depth=depth, relu=True)

        self.P.attn_token = Relation(token_dim, 1, out_channels, out_channels)  # t d, b n o -> b t o
        self.P.reattn_token = block(out_channels, heads, downsample=True)  # b t o, b n o -> b t o
        self.P.attn = nn.Sequential(*[block(out_channels, heads)
                                      for _ in range(depth - 1)])  # b t o, b t o -> b t o

        self.attn = self.P

        if ViRS:
            self.attn = nn.Sequential(self.P.reattn_token, self.P.attn)

    def repr_shape(self, c, h, w):
        if self.ViRS:
            return super().repr_shape(c, h, w)
        return self.out_channels, self.tokens, 1


# MHDPR
class Relation(nn.Module):
    def __init__(self, x_dim=32, heads=None, s_dim=None, v_dim=None, talk_h=False, single_h_tokens=False):
        super().__init__()

        s_dim = x_dim if s_dim is None \
            else s_dim

        v_dim = x_dim if v_dim is None \
            else v_dim

        heads = math.gcd(8, v_dim) if heads is None \
            else heads

        self.x_dim = x_dim
        self.v_dim = v_dim
        self.heads = heads

        assert v_dim % heads == 0, f'value dim={x_dim} is not divisible by heads={heads}'

        self.to_q = nn.Linear(x_dim, x_dim, bias=False)
        k_dim = x_dim * self.heads + v_dim if single_h_tokens else x_dim + v_dim
        self.to_kv = nn.Linear(s_dim, k_dim, bias=False)

        # "Talking heads" (https://arxiv.org/abs/2003.02436)
        self.talk_h = nn.Sequential(Utils.ChSwap, nn.Linear(heads, heads, bias=False),
                                    nn.LayerNorm(heads), Utils.ChSwap) if talk_h else nn.Identity()

    def forward(self, x, s=None):
        # Conserves shape
        shape = x.shape
        assert shape[-1] == self.x_dim, f'input dim ≠ pre-specified {shape[-1]}≠{self.x_dim}'

        if s is None:
            s = x

        tokens = len(x.shape) == 2  # Tokens distinguished by having axes=2
        if not tokens:
            x = x.flatten(1, -2)
        s = s.flatten(1, -2)

        q = x if tokens else self.to_q(x)
        k, v = self.to_kv(s).tensor_split([self.x_dim], dim=-1)

        multi_head_tokens = q.shape[-1] == k.shape[-1] and tokens

        if multi_head_tokens or not tokens:
            pattern = 'n (h d) -> h n d' if tokens \
                else 'b n (h d) -> b h n d'
            q = rearrange(q, pattern, h=self.heads)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (k, v))

        # Memory efficient toggle, e.g., =0.5
        mem_limit = False
        einsum = EinsumPlanner(q.device, cuda_mem_limit=mem_limit).einsum if 0 < mem_limit < 1 \
            else torch.einsum

        pattern = 'h i d, b h j d -> b h i j' if multi_head_tokens \
            else 'i d, b h j d -> b h i j' if tokens \
            else 'b h i d, b h j d -> b h i j'
        self.dots = einsum(pattern, q, k) * self.x_dim ** -0.5

        weights = self.dots.softmax(dim=-1)

        if 0 < mem_limit < 1:
            weights = weights.to(q.device)

        # "Talking heads"
        weights = self.talk_h(weights)

        attn = einsum('b h i j, b h j d -> b h i d', weights, v)

        if 0 < mem_limit < 1:
            attn = attn.to(q.device)

        rtn = torch.argmax(weights, dim=-1)  # [b, h, i]
        rtn = Utils.gather_indices(v, rtn, dim=-2)  # [b, h, i, d]
        rtn = attn - (attn - rtn).detach()

        out = rearrange(rtn, 'b h n d -> b n (h d)')

        # Restores original shape
        if not tokens:
            out = out.view(*shape[:-1], -1)

        return out


# Concat, +residual from input
class ConcatBlock(nn.Module):
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


# In-to-mid residual, concat, mid-to-out residual
class CourseCorrectorBlock(ConcatBlock):
    def forward(self, x, context=None):
        if context is None:
            context = x

        attn = self.LN_mid(self.attn(x, context)) + x  # Attention = Reason-er
        out = self.LN_out(self.mlp(attn, x)) + attn  # MLP = Course Corrector

        return out


# Heads layer norm'd
class Disentangled(ConcatBlock):
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


# Head, in
class IndependentHeadsBlock(ConcatBlock):
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


# Head, head:in
class RelativeBlock(ConcatBlock):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None, downsample=False):
        super().__init__(dim, heads, context_dim, value_dim)

        self.attn = ReLA(dim, self.heads, self.context_dim, self.value_dim * self.heads)
        self.RN = RN(dim, dim * 2, inner_depth=0, outer_depth=0, mid_nonlinearity=nn.ReLU(inplace=True))

        self.downsample = nn.Linear(dim, self.value_dim) if downsample else nn.Identity()

    def forward(self, x, context=None):
        if context is None:
            context = x  # [b, n, d] or [n, d] if tokens

        shape = x.shape

        attn = self.attn(x, context)  # [b, n, h * d]
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)  # [b, n, h, d]

        norm = self.LN_mid(head_wise)  # [b, n, h, d]
        residual = x.unsqueeze(-2)  # [b, n, 1, d]

        relation = norm.flatten(0, -3)  # [b * n, h, d]
        residual = residual.flatten(0, -3)  # [b * n, 1, d]
        context = torch.cat([residual.expand(*relation.shape[:-1], -1), relation], -1)  # [b * n, h, d * 2]

        out = self.LN_out(self.RN(relation, context))  # [b * n, d]

        return out.view(*(shape[:-2] or [-1]), *shape[-2:]) + self.downsample(x)  # [b, n, d]


# Re-param
class RelationBlock(RelativeBlock):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None, downsample=False):
        super().__init__(dim, heads, context_dim, value_dim, downsample)

        self.attn = Relation(dim, self.heads, self.context_dim, self.value_dim * self.heads)

