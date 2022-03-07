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
from Blocks.Architectures.MultiHeadAttention import ReLA, mem_efficient_attend
from Blocks.Architectures.RN import RN
from Blocks.Architectures.Perceiver import Perceiver, PerceiverV2


class ViRPV2(ViT):
    def __init__(self, input_shape, patch_size=4, out_channels=128, emb_dropout=0, tokens=20, token_dim=128,
                 qk_dim=None, v_dim=None, hidden_dim=None, heads=8, depths=[8], recursions=None, dropout=0,
                 pool='cls', output_dim=None, experiment='relation', ViRS=False):
        self.tokens = tokens
        self.ViRS = ViRS

        token_dim = out_channels if token_dim is None else token_dim
        v_dim = out_channels if v_dim is None else v_dim

        depths = [3] if depths is None else depths
        recursions = [1 for _ in depths] if recursions is None else recursions

        assert len(depths) == len(recursions), 'Recursion must be specified for each depth'
        assert token_dim == v_dim or recursions[0] == 1, 'First depth cannot be recursive if token_dim ≠ value_dim'

        super().__init__(input_shape, patch_size, out_channels, emb_dropout, pool=pool, output_dim=output_dim)

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

        attn = nn.ModuleList([nn.Identity()] +
                             sum([[nn.Sequential(*[block(out_channels, heads,
                                                         qk_dim=qk_dim, v_dim=v_dim, hidden_dim=hidden_dim,
                                                         dropout=dropout)
                                                   for _ in range(inner_depth - 1)])] * recurs
                                  for recurs, inner_depth in zip(recursions, depths)], []))

        if ViRS:
            self.P = attn
        else:
            self.P = PerceiverV2(out_channels, heads, tokens, token_dim, fix_token=True)

            self.P.reattn = nn.ModuleList(([Relation(token_dim, 1, out_channels, qk_dim, out_channels)]) +
                                          sum([[block(token_dim if i == 0 else out_channels, heads,
                                                      qk_dim=qk_dim, v_dim=v_dim, hidden_dim=hidden_dim,
                                                      dropout=dropout)] * recurs
                                               for i, recurs in enumerate(recursions)], []))
            self.P.attn = attn

        self.attn = self.P

        if ViRS:
            # Assumes only 1 depth
            self.attn = nn.Sequential(self.P.attn[1])

    def repr_shape(self, c, h, w):
        if self.ViRS:
            return super().repr_shape(c, h, w)
        return self.out_channels, self.tokens, 1


class ViRP(ViT):
    def __init__(self, input_shape, patch_size=4, out_channels=128, heads=8, tokens=64,
                 token_dim=128, depth=8, pool='cls', output_dim=None, experiment='relation', ViRS=True):
        self.tokens = tokens
        self.ViRS = ViRS  # ViRP? Visiorelational Perceptor

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
        # self.P.attn_token = block(token_dim, heads, out_channels, out_channels)  # t d, b n o -> b t o
        self.P.reattn_token = block(out_channels, heads)  # b t o, b n o -> b t o
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
    def __init__(self, dim=32, heads=None, s_dim=None, qk_dim=None, v_dim=None, talk_h=False):
        super().__init__()

        self.dim = dim

        s_dim = dim if s_dim is None else s_dim
        qk_dim = dim if qk_dim is None else qk_dim
        v_dim = dim if v_dim is None else v_dim

        heads = math.gcd(8, v_dim) if heads is None \
            else heads

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.heads = heads

        assert v_dim % heads == 0, f'value dim={v_dim} is not divisible by heads={heads}'

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_kv = nn.Linear(s_dim, qk_dim + v_dim, bias=False)

        # "Talking heads" (https://arxiv.org/abs/2003.02436)
        self.talk_h = nn.Sequential(Utils.ChSwap, nn.Linear(heads, heads, bias=False),
                                    nn.LayerNorm(heads), Utils.ChSwap) if talk_h else nn.Identity()

    def forward(self, x, s=None):
        # Conserves shape
        shape = x.shape
        assert shape[-1] == self.dim, f'input dim ≠ pre-specified {shape[-1]}≠{self.dim}'

        if s is None:
            s = x

        tokens = len(x.shape) == 2  # Tokens distinguished by having axes=2
        if not tokens:
            x = x.flatten(1, -2)
        s = s.flatten(1, -2)

        q = x if tokens else self.to_q(x)
        k, v = self.to_kv(s).tensor_split([self.qk_dim], dim=-1)

        multi_head_tokens = q.shape[-1] == k.shape[-1] and tokens

        assert q.shape[-1] == k.shape[-1] / self.heads or not tokens, \
            f'Tokens, keys cannot be broadcast {q.shape[-1]}, {k.shape[-1]}'

        if multi_head_tokens or not tokens:
            pattern = 'n (h d) -> h n d' if tokens \
                else 'b n (h d) -> b h n d'
            q = rearrange(q, pattern, h=self.heads)

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (k, v))

        # Memory limit toggle, e.g., =0.5
        mem_limit = False
        einsum = EinsumPlanner(q.device, cuda_mem_limit=mem_limit).einsum if 0 < mem_limit < 1 \
            else torch.einsum

        scale = q.shape[-1] ** -0.5
        q = q * scale

        pattern = 'h i d, b h j d -> b h i j' if multi_head_tokens \
            else 'i d, b h j d -> b h i j' if tokens \
            else 'b h i d, b h j d -> b h i j'

        # Memory efficient toggle
        mem_efficient = False
        if mem_efficient:
            attn, weights = mem_efficient_attend(q, k, v, pattern=pattern)
        else:
            self.weights = einsum(pattern, q, k)
            # self.dots = self.dots - self.dots.amax(dim=-1, keepdim=True).detach()

            weights = self.weights.softmax(dim=-1)

            if 0 < mem_limit < 1:
                weights = weights.to(q.device)

            # "Talking heads"
            weights = self.talk_h(weights)

            # attn = torch.einsum('b h i j, b h j d -> b h i d', weights, v)
            attn = torch.matmul(weights, v)

        rtn = torch.argmax(weights, dim=-1)  # [b, h, i]
        rtn = Utils.gather_indices(v, rtn, dim=-2)  # [b, h, i, d]
        rtn = attn - (attn - rtn).detach()

        out = rearrange(rtn, 'b h n d -> b n (h d)')

        # Restores original leading dims
        if not tokens:
            out = out.view(*shape[:-1], -1)

        if 0 < mem_limit < 1:
            out = out.to(q.device)

        return out


# Concat, +residual from input
class ConcatBlock(nn.Module):
    def __init__(self, dim=32, heads=8, s_dim=None, qk_dim=None, v_dim=None, hidden_dim=None, dropout=0):
        super().__init__()

        v_dim = dim if v_dim is None else v_dim

        self.heads = math.gcd(8, v_dim) if heads is None else heads

        self.s_dim = s_dim
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        hidden_dim = v_dim * 4 if hidden_dim is None else hidden_dim

        self.attn = ReLA(dim, self.heads, s_dim, qk_dim, v_dim)
        self.mlp = nn.Sequential(MLP(v_dim, v_dim, hidden_dim, 1, nn.GELU(), dropout), nn.Dropout(dropout))

        self.LN_mid = nn.LayerNorm(v_dim)
        self.LN_out = nn.LayerNorm(v_dim)

    def repr_shape(self, c, h, w):
        return self.v_dim, h, w  # Assumes channels last

    def forward(self, x, context=None):
        if context is None:
            context = x

        attn = self.LN_mid(self.attn(x, context))  # Relation
        out = self.LN_out(self.mlp(attn, x)) + x  # Reason-er

        return out


# In-to-mid residual, concat, mid-to-out residual
class CourseCorrectorBlock(ConcatBlock):
    def forward(self, x, s=None):
        if s is None:
            s = x

        attn = self.LN_mid(self.attn(x, s)) + x  # In this block, Attention = Reason-er,
        out = self.LN_out(self.mlp(attn, x)) + attn  # MLP = Course Corrector

        return out


# Heads layer norm'd
class Disentangled(ConcatBlock):
    def __init__(self, dim=32, heads=8, s_dim=None, qk_dim=None, v_dim=None, hidden_dim=None, dropout=0):
        super().__init__(dim, heads, s_dim, qk_dim, v_dim, hidden_dim, dropout)

        self.LN_mid = nn.LayerNorm(self.v_dim // self.heads)

    def forward(self, x, s=None):
        if s is None:
            s = x

        attn = self.attn(x, s)
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)
        norm = self.LN_mid(head_wise)
        disentangled = norm.view(attn.shape)

        out = self.LN_out(self.mlp(disentangled, x)) + x

        return out


# Head, in
class IndependentHeadsBlock(ConcatBlock):
    def __init__(self, dim=32, heads=1, s_dim=None, qk_dim=None, v_dim=None, hidden_dim=None, dropout=0):
        super().__init__(dim, heads, s_dim, qk_dim, v_dim, hidden_dim, dropout)

        self.attn = ReLA(dim, self.heads, self.s_dim, self.qk_dim, self.v_dim * self.heads)
        self.RN = RN(dim, dim, 0, 0, hidden_dim)

    def forward(self, x, s=None):
        if s is None:
            s = x  # [b, n, d]

        attn = self.attn(x, s)  # [b, n, h * d]
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)  # [b, n, h, d]

        norm = self.LN_mid(head_wise)  # [b, n, h, d]
        residual = x.unsqueeze(-2)  # [b, n, 1, d]

        relation = norm.flatten(0, -3)  # [b * n, h, d]
        residual = residual.flatten(0, -3)  # [b * n, 1, d]

        out = self.LN_out(self.RN(relation, residual))  # [b * n, d]

        return out.view(x.shape) + x  # [b, n, d]


# Head, head:in
class RelativeBlock(ConcatBlock):
    def __init__(self, dim=32, heads=1, s_dim=None, qk_dim=None, v_dim=None, hidden_dim=None, dropout=0):
        super().__init__(dim, heads, s_dim, qk_dim, v_dim, hidden_dim, dropout)

        self.attn = ReLA(dim, self.heads, self.s_dim, self.qk_dim, self.v_dim * self.heads)
        self.RN = RN(dim, dim * 2, 0, 0, hidden_dim, mid_nonlinearity=nn.GELU(), dropout=dropout)

        self.downsample = nn.Linear(dim, self.v_dim) if dim != self.v_dim \
            else nn.Identity()

    def forward(self, x, s=None):
        if s is None:
            s = x  # [b, n, d] or [n, d] if tokens

        shape = x.shape

        attn = self.attn(x, s)  # [b, n, h * d]
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)  # [b, n, h, d]

        residual = x.unsqueeze(-2)  # [b, n, 1, d] or [n, 1, d] if tokens
        residual = residual.expand(*head_wise.shape[:-1], -1)  # [b, n, h, d]
        residual = residual.flatten(0, -3)  # [b * n, h, d]

        norm = self.LN_mid(head_wise)  # [b, n, h, d]
        relation = norm.flatten(0, -3)  # [b * n, h, d]

        s = torch.cat([residual, relation], -1)  # [b * n, h, d * 2]

        out = self.LN_out(self.RN(relation, s))  # [b * n, d]

        return out.view(*(shape[:-2] or [-1]), *shape[-2:]) + self.downsample(x)  # [b, n, d]


# Re-param
class RelationBlock(RelativeBlock):
    def __init__(self, dim=32, heads=1, s_dim=None, qk_dim=None, v_dim=None, hidden_dim=None, dropout=0):
        super().__init__(dim, heads, s_dim, qk_dim, v_dim, hidden_dim, dropout)

        self.attn = Relation(dim, self.heads, self.s_dim, self.qk_dim, self.v_dim * self.heads)

