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
from Blocks.Architectures.MultiHeadAttention import ReLA, mem_efficient_attend, SelfAttentionBlock, SelfAttention
from Blocks.Architectures.Perceiver import Perceiver
from Blocks.Architectures.RN import RN


class ViRP(ViT):
    """Visiorelational Perceptor"""
    def __init__(self, input_shape, patch_size=4, out_channels=128, emb_dropout=0, tokens=20, token_dim=128,
                 k_dim=None, v_dim=None, hidden_dim=None, heads=8, depth=6, depths=None, recursions=None, dropout=0,
                 pool_type='cls', output_dim=None, experiment='pairwise_relation', perceiver=False):
        self.tokens = tokens
        self.perceiver = perceiver

        token_dim = out_channels if token_dim is None else token_dim
        v_dim = out_channels if v_dim is None else v_dim

        depths = [depth] if depths is None else depths
        recursions = [1 for _ in depths] if recursions is None else recursions

        assert len(depths) == len(recursions), 'Recursion must be specified for each depth'
        assert token_dim == v_dim or recursions[0] == 1, 'First depth cannot be recursive if token_dim ≠ value_dim'

        super().__init__(input_shape, patch_size, out_channels, emb_dropout, pool_type=pool_type, output_dim=output_dim)

        kwargs = {}
        if experiment == 'pairwise_relation':
            block = PairwiseRelationBlock
            kwargs = dict(impartial_q_head=False)
        elif experiment == 'impartial_relation':
            block = ImpartialRelationBlock
        elif experiment == 'relation':
            block = RelationBlock
        elif experiment == 'relative':
            block = RelativeBlock
        elif experiment == 'pairwise':
            block = PairwiseHeadsBlock
        elif experiment == 'independent':
            block = IndependentHeadsBlock
        elif experiment == 'disentangled':
            block = DisentangledBlock
        elif experiment == 'course_corrector_norm':
            block = CourseCorrectorNormBlock
        elif experiment == 'course_corrector':
            block = CourseCorrectorBlock
        elif experiment == 'no_norm':
            block = ConcatNoNormBlock
        elif experiment == 'pure_pre_norm_once':
            block = ConcatPurePreNormOnceBlock
        elif experiment == 'pure_pre_norm':
            block = ConcatPurePreNormBlock
        elif experiment == 'pre_norm':
            block = ConcatPreNormBlock
        elif experiment == 'pre_norm_once':
            block = ConcatPreNormOnceBlock
        elif experiment == 'concat':
            block = ConcatBlock
        else:
            raise NotImplementedError('No such experiment')

        attn = nn.ModuleList([nn.Identity()] +
                             sum([[nn.Sequential(*[block(out_channels, heads,
                                                         k_dim=k_dim, v_dim=v_dim, hidden_dim=hidden_dim,
                                                         dropout=dropout, **kwargs)
                                                   for _ in range(inner_depth - int(perceiver))])] * recurs
                                  for recurs, inner_depth in zip(recursions, depths)], []))

        if perceiver:
            self.attn = Perceiver(out_channels, heads, tokens, token_dim, fix_token=True)

            self.attn.reattn = nn.ModuleList(([Relation(token_dim, 1, out_channels, k_dim, out_channels)]) +
                                             sum([[block(token_dim if i == 0 else out_channels, heads,
                                                         k_dim=k_dim, v_dim=v_dim, hidden_dim=hidden_dim,
                                                         dropout=dropout, **kwargs)] * recurs
                                                  for i, recurs in enumerate(recursions)], []))
            self.attn.attn = attn
        else:
            self.attn = nn.Sequential(attn[1])

    def repr_shape(self, c, h, w):
        if self.perceiver:
            return self.out_channels, self.tokens, 1
        return super().repr_shape(c, h, w)


# MHDPR
class Relation(nn.Module):
    """Multi-Head Dot-Product Relation"""
    def __init__(self, dim=32, heads=None, s_dim=None, k_dim=None, v_dim=None, talk_h=False, impartial_q_head=False):
        super().__init__()

        self.dim = dim

        s_dim = dim if s_dim is None else s_dim
        k_dim = dim if k_dim is None else k_dim
        v_dim = dim if v_dim is None else v_dim

        heads = math.gcd(8, v_dim) if heads is None \
            else heads

        self.k_dim = k_dim
        self.v_dim = v_dim
        self.heads = heads

        assert v_dim % heads == 0, f'value dim={v_dim} is not divisible by heads={heads}'

        self.to_q = nn.Linear(dim, k_dim // heads if impartial_q_head else k_dim, bias=False)
        self.to_kv = nn.Linear(s_dim, k_dim + v_dim, bias=False)

        # "Talking heads" (https://arxiv.org/abs/2003.02436)
        self.talk_h = nn.Sequential(Utils.ChSwap, nn.Linear(heads, heads, bias=False),
                                    nn.LayerNorm(heads), Utils.ChSwap) if talk_h else nn.Identity()

    def forward(self, x, s=None):
        # Conserves shape
        shape = x.shape
        assert shape[-1] == self.dim, f'input dim ≠ pre-specified {shape[-1]}≠{self.dim}'

        if s is None:
            s = x

        tokens = len(x.shape) == 2  # Tokens distinguished by having axes=2 (no batch dim)
        if not tokens:
            x = x.flatten(1, -2)
        s = s.flatten(1, -2)

        q = x if tokens else self.to_q(x)
        k, v = self.to_kv(s).tensor_split([self.k_dim], dim=-1)

        multi_head_q = q.shape[-1] == k.shape[-1]

        assert q.shape[-1] == k.shape[-1] / self.heads or multi_head_q, \
            f'Queries, keys cannot be broadcast {q.shape[-1]}, {k.shape[-1]}'

        if multi_head_q:
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

        pattern = 'h i d, b h j d -> b h i j' if multi_head_q and tokens \
            else 'i d, b h j d -> b h i j' if tokens \
            else 'b h i d, b h j d -> b h i j' if multi_head_q \
            else 'b i d, b h j d -> b h i j'

        # Memory efficient toggle
        mem_efficient = False
        if mem_efficient:
            attn, weights = mem_efficient_attend(q, k, v, pattern=pattern)
        else:
            self.weights = weights = einsum(pattern, q, k)
            # self.dots = self.dots - self.dots.amax(dim=-1, keepdim=True).detach()

            if self.training:
                weights = self.weights.softmax(dim=-1)

                if 0 < mem_limit < 1:
                    weights = weights.to(q.device)

                # "Talking heads"
                weights = self.talk_h(weights)

                # attn = torch.einsum('b h i j, b h j d -> b h i d', weights, v)
                attn = torch.matmul(weights, v)

        rtn = torch.argmax(weights, dim=-1)  # [b, h, i]
        rtn = Utils.gather_indices(v, rtn, dim=-2)  # [b, h, i, d]

        if self.training:
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
    def __init__(self, dim=32, heads=8, s_dim=None, k_dim=None, v_dim=None, hidden_dim=None, dropout=0):
        super().__init__()

        v_dim = dim if v_dim is None else v_dim
        hidden_dim = v_dim * 4 if hidden_dim is None else hidden_dim

        self.heads = math.gcd(8, v_dim) if heads is None else heads

        self.s_dim = s_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.hidden_dim = hidden_dim

        self.attn = SelfAttention(dim, self.heads, s_dim, k_dim, v_dim)
        # self.attn = ReLA(dim, self.heads, s_dim, k_dim, v_dim)  # TODO SelfAttention here, make separate Relative
        # self.project = nn.Identity() if heads == 1 \
        #     else nn.Sequential(nn.Linear(v_dim, dim), nn.Dropout(dropout))
        self.mlp = MLP(v_dim + dim, dim, hidden_dim, 1, nn.GELU(), dropout)
        self.dropout = nn.Dropout(dropout)

        self.LN_mid = nn.LayerNorm(v_dim)  # dim if project
        self.LN_out = nn.LayerNorm(dim)

    def repr_shape(self, c, h, w):
        return self.v_dim, h, w  # Assumes channels last

    def forward(self, x, s=None):
        if s is None:
            s = x

        attn = self.LN_mid(self.attn(x, s))  # Relation
        out = self.LN_out(self.dropout(self.mlp(attn, x))) + x  # Reason-er

        return out


class ConcatPreNormOnceBlock(ConcatBlock):
    def forward(self, x, s=None):
        pre_norm = self.LN_out(x)

        if s is None:
            s = pre_norm

        attn = self.LN_mid(self.attn(pre_norm, s))  # Relation
        out = self.dropout(self.mlp(attn, x)) + x  # Reason-er

        return out


class ConcatPreNormBlock(ConcatBlock):
    def forward(self, x, s=None):
        pre_norm = self.LN_out(x)

        if s is None:
            s = pre_norm

        attn = self.LN_mid(self.attn(pre_norm, s))  # Relation  TODO Disentangled head-wise LN
        out = self.dropout(self.mlp(attn, pre_norm)) + x  # Reason-er  TODO Alternatively, a new LN instead of pre_norm

        return out


class ConcatPurePreNormBlock(ConcatBlock):
    def forward(self, x, s=None):
        pre_norm = self.LN_out(x)

        if s is None:
            s = pre_norm

        attn = self.attn(pre_norm, s)  # Relation
        out = self.dropout(self.mlp(attn, pre_norm)) + x  # Reason-er

        return out


class ConcatPurePreNormOnceBlock(ConcatBlock):
    def forward(self, x, s=None):
        pre_norm = self.LN_out(x)

        if s is None:
            s = pre_norm

        attn = self.attn(pre_norm, s)  # Relation
        out = self.dropout(self.mlp(attn, x)) + x  # Reason-er

        return out


class ConcatNoNormBlock(ConcatBlock):
    def forward(self, x, s=None):
        if s is None:
            s = x

        attn = self.attn(x, s)  # Relation
        out = self.dropout(self.mlp(attn, x)) + x  # Reason-er

        return out


# TODO Unify pre_norm method


# In-to-mid residual, concat, mid-to-out residual
class CourseCorrectorBlock(ConcatBlock):  # "PurePreNormOnce"
    def forward(self, x, s=None):
        pre_norm = self.LN_out(x)
        
        if s is None:
            s = pre_norm

        attn = self.attn(pre_norm, s) + x  # In this block, Attention = Reason-er,
        out = self.mlp(attn, x) + attn  # MLP = Course Corrector

        return out


# In-to-mid residual, concat, mid-to-out residual
class CourseCorrectorNormBlock(ConcatBlock):  # "PreNormOnce"
    def forward(self, x, s=None):
        pre_norm = self.LN_out(x)

        if s is None:
            s = pre_norm

        attn = self.attn(pre_norm, s) + x  # In this block, Attention = Reason-er,
        out = self.mlp(self.LN_mid(attn), x) + attn  # MLP = Course Corrector

        return out


# Heads layer norm'd
class DisentangledBlock(ConcatBlock):
    def __init__(self, dim=32, heads=8, s_dim=None, k_dim=None, v_dim=None, hidden_dim=None, dropout=0):
        super().__init__(dim, heads, s_dim, k_dim, v_dim, hidden_dim, dropout)

        # self.project = nn.Identity() if heads == 1 \
        #     else nn.Sequential(nn.Linear(self.v_dim // self.heads, self.v_dim // self.heads), nn.Dropout(dropout))
        self.LN_mid = nn.LayerNorm(self.v_dim // self.heads)

    def forward(self, x, s=None):
        if s is None:
            s = x

        attn = self.attn(x, s)
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)
        norm = self.LN_mid(head_wise)  # Can apply self.project before every LN_mid
        disentangled = norm.view(attn.shape)

        out = self.LN_out(self.dropout(self.mlp(disentangled, x))) + x

        return out


# Head, in
class IndependentHeadsBlock(DisentangledBlock):
    def __init__(self, dim=32, heads=1, s_dim=None, k_dim=None, v_dim=None, hidden_dim=None, dropout=0):
        super().__init__(dim, heads, s_dim, k_dim, v_dim, hidden_dim, dropout)

        self.downsample_mid = nn.Linear(dim, self.v_dim // self.heads) if dim != self.v_dim // self.heads \
            else nn.Identity()

        self.RN = RN(self.v_dim // self.heads, self.v_dim // self.heads, 0, 0, self.hidden_dim, self.v_dim,
                     mid_nonlinearity=nn.GELU(), dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.downsample_out = nn.Linear(dim, self.v_dim) if dim != self.v_dim \
            else nn.Identity()

    def forward(self, x, s=None):
        if s is None:
            s = x  # [b, n, d] or [n, d] if tokens

        shape = x.shape

        attn = self.attn(x, s)  # [b, n, h * d]
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)  # [b, n, h, d]

        residual = self.downsample_mid(x)  # [b * n, 1, d] or [n, 1, d] if tokens
        residual = residual.unsqueeze(-2)  # [b, n, 1, d] or [n, 1, d] if tokens
        residual = residual.expand(shape[0], -1, -1, -1)  # [b, n, 1, d]
        residual = residual.view(-1, 1, residual.shape[-1])  # [b * n, 1, d]

        norm = self.LN_mid(head_wise)  # [b, n, h, d]
        relation = norm.view(-1, *norm.shape[-2:])  # [b * n, h, d]

        out = self.LN_out(self.dropout(self.RN(relation, residual)))  # [b * n, d]

        return out.view(*(shape[:-2] or [-1]), *shape[-2:]) + self.downsample_out(x)  # [b, n, d]


# Head, head:in
class PairwiseHeadsBlock(IndependentHeadsBlock):
    def __init__(self, dim=32, heads=1, s_dim=None, k_dim=None, v_dim=None, hidden_dim=None, dropout=0):
        super().__init__(dim, heads, s_dim, k_dim, v_dim, hidden_dim, dropout)

        self.RN = RN(self.v_dim // self.heads, 2 * self.v_dim // self.heads, 0, 0, self.hidden_dim, self.v_dim,
                     mid_nonlinearity=nn.GELU(), dropout=dropout)

    def forward(self, x, s=None):
        if s is None:
            s = x  # [b, n, d] or [n, d] if tokens

        shape = x.shape

        attn = self.attn(x, s)  # [b, n, h * d]
        head_wise = attn.view(*attn.shape[:-1], self.heads, -1)  # [b, n, h, d]

        residual = self.downsample_mid(x)  # [b * n, 1, d] or [n, 1, d] if tokens
        residual = residual.unsqueeze(-2)  # [b, n, 1, d] or [n, 1, d] if tokens
        residual = residual.expand(*head_wise.shape[:-1], -1)  # [b, n, h, d]
        residual = residual.view(-1, *residual.shape[-2:])  # [b * n, h, d]

        norm = self.LN_mid(head_wise)  # [b, n, h, d]
        relation = norm.view(-1, *norm.shape[-2:])  # [b * n, h, d]

        s = torch.cat([residual, relation], -1)  # [b * n, h, d * 2]

        out = self.LN_out(self.dropout(self.RN(relation, s)))  # [b * n, d]

        return out.view(*(shape[:-2] or [-1]), *shape[-2:]) + self.downsample_out(x)  # [b, n, d]


# Rectified-linear attention
class RelativeBlock(IndependentHeadsBlock):
    def __init__(self, dim=32, heads=1, s_dim=None, k_dim=None, v_dim=None, hidden_dim=None, dropout=0):
        super().__init__(dim, heads, s_dim, k_dim, v_dim, hidden_dim, dropout)

        self.attn = ReLA(dim, self.heads, self.s_dim, self.k_dim, self.v_dim)


# Re-param
class RelationBlock(IndependentHeadsBlock):
    def __init__(self, dim=32, heads=1, s_dim=None, k_dim=None, v_dim=None, hidden_dim=None, dropout=0):
        super().__init__(dim, heads, s_dim, k_dim, v_dim, hidden_dim, dropout)

        self.attn = Relation(dim, self.heads, self.s_dim, self.k_dim, self.v_dim)


# Re-param
class ImpartialRelationBlock(IndependentHeadsBlock):
    def __init__(self, dim=32, heads=1, s_dim=None, k_dim=None, v_dim=None, hidden_dim=None, dropout=0):
        super().__init__(dim, heads, s_dim, k_dim, v_dim, hidden_dim, dropout)

        self.attn = Relation(dim, self.heads, self.s_dim, self.k_dim, self.v_dim, impartial_q_head=True)


# Re-param
class PairwiseRelationBlock(PairwiseHeadsBlock):
    def __init__(self, dim=32, heads=1, s_dim=None, k_dim=None, v_dim=None, hidden_dim=None, dropout=0,
                 impartial_q_head=False):
        super().__init__(dim, heads, s_dim, k_dim, v_dim, hidden_dim, dropout)

        self.attn = Relation(dim, self.heads, self.s_dim, self.k_dim, self.v_dim, impartial_q_head=impartial_q_head)

