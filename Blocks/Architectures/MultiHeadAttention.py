# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

from einops import rearrange, repeat

import torch
from torch import nn

from Blocks.Architectures.MLP import MLP
from Blocks.Architectures.Vision.CNN import AvgPool

import Utils


class CrossAttention(nn.Module):
    """
    Multi-head dot-product attention
    (https://arxiv.org/abs/1706.03762?context=cs)
    """
    def __init__(self, dim=32, heads=None, context_dim=None, qk_dim=None, v_dim=None, talk_h=False, rela=False):
        super().__init__()

        self.dim = dim

        context_dim = dim if context_dim is None else context_dim
        qk_dim = dim if qk_dim is None else qk_dim
        v_dim = dim if v_dim is None else v_dim

        heads = math.gcd(8, v_dim) if heads is None \
            else heads

        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.heads = heads

        assert v_dim % heads == 0, f'value dim={dim} is not divisible by heads={heads}'

        self.weights = None  # Can access attention weights

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, qk_dim + v_dim, bias=False)

        # "Talking heads" (https://arxiv.org/abs/2003.02436)
        self.talk_h = nn.Sequential(Utils.ChSwap, nn.Linear(heads, heads, bias=False),
                                    nn.LayerNorm(heads), Utils.ChSwap) if talk_h else nn.Identity()

        self.relu = nn.ReLU(inplace=True) if rela else None  # ReLA (https://arxiv.org/abs/2104.07012)

    def forward(self, x, context=None):
        # Conserves shape
        shape = x.shape
        assert shape[-1] == self.dim, f'input dim ≠ pre-specified {shape[-1]}≠{self.dim}'

        if context is None:
            context = x

        tokens = len(x.shape) == 2  # Tokens (e.g. for Perceiver) distinguished by having axes=2 (no batch dim)
        if not tokens:
            x = x.flatten(1, -2)  # Assumes channels-last
        context = context.flatten(1, -2)  # Assumes channels-last

        q = x if tokens else self.to_q(x)
        k, v = self.to_kv(context).tensor_split([self.qk_dim], dim=-1)

        if tokens:
            assert q.shape[-1] == k.shape[-1] / self.heads, f'Tokens, keys cannot broadcast {q.shape[-1]}≠{k.shape[-1]}'

        multi_head_tokens = q.shape[-1] == k.shape[-1] and tokens  # In case tokens themselves have multiple heads

        if multi_head_tokens or not tokens:
            pattern = 'n (h d) -> h n d' if tokens \
                else 'b n (h d) -> b h n d'
            q = rearrange(q, pattern, h=self.heads)  # Heads-first

        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (k, v))

        scale = q.shape[-1] ** -0.5
        q = q * scale

        pattern = 'h i d, b h j d -> b h i j' if multi_head_tokens \
            else 'i d, b h j d -> b h i j' if tokens \
            else 'b h i d, b h j d -> b h i j'

        self.weights = torch.einsum(pattern, q, k)
        # self.weights = self.weights - self.weights.amax(dim=-1, keepdim=True).detach()

        weights = self.weights.softmax(dim=-1) if self.relu is None \
            else self.relu(self.weights)

        # "Talking heads"
        weights = self.talk_h(weights)

        attn = torch.matmul(weights, v)

        out = rearrange(attn, 'b h n d -> b n (h d)')

        # Restores original leading dims
        if not tokens:
            out = out.view(*shape[:-1], -1)

        return out


# A minimalist implementation of the above using only Pytorch natives
class CrossAttend(nn.Module):
    def __init__(self, dim=32, heads=8, context_dim=None, v_dim=None, *_):
        super().__init__()

        self.weights = None
        self.attn = nn.MultiheadAttention(dim, heads, kdim=context_dim, vdim=v_dim, batch_first=True)

    def forward(self, x, context):
        # Conserves shape
        mid_shape = x.shape[1:-1]

        x = x.flatten(1, -2)
        context = context.flatten(1, -2)

        attn, self.weights = self.attn(x, context, context)

        # Restores original shape
        return attn.view(-1, *mid_shape, attn.shape[-1])


class ReLA(CrossAttention):
    """ReLA: Rectified linear attention (https://arxiv.org/abs/2104.07012)"""
    def __init__(self, dim=32, heads=None, context_dim=None, qk_dim=None, v_dim=None):
        super().__init__(dim, heads, context_dim, qk_dim, v_dim, False, True)


class SelfAttention(CrossAttention):
    """Self-attention, just cross-attention except where context = input"""
    def forward(self, x, *_):
        return super().forward(x)


class CrossAttentionBlock(nn.Module):
    """A Transformer pre-norm block, but for arbitrary context
    (https://arxiv.org/pdf/2002.04745.pdf)"""
    def __init__(self, dim=32, heads=None, context_dim=None, qk_dim=None, v_dim=None, hidden_dim=None, dropout=0,
                 talk_h=False, rela=False):
        super().__init__()

        v_dim = dim if v_dim is None else v_dim
        hidden_dim = v_dim * 4 if hidden_dim is None else hidden_dim

        self.heads = math.gcd(8, v_dim) if heads is None else heads

        self.v_dim = v_dim

        self.attn = CrossAttention(dim, self.heads, context_dim, qk_dim, v_dim, talk_h, rela)
        self.LN_ReLA = nn.LayerNorm(v_dim) if rela \
            else nn.Identity()
        self.project = nn.Identity() if heads == 1 \
            else nn.Sequential(nn.Linear(v_dim, dim), nn.Dropout(dropout))
        self.mlp = nn.Sequential(MLP(dim, dim, hidden_dim, 1, nn.GELU(), dropout), nn.Dropout(dropout))

        self.LN_pre = nn.LayerNorm(dim)
        self.LN_mid = nn.LayerNorm(dim)

    def repr_shape(self, c, h, w):
        return self.v_dim, h, w  # Assumes channels last

    def forward(self, x, context=None):
        pre_norm = self.LN_pre(x)

        if context is None:
            context = pre_norm

        attn = self.project(self.LN_ReLA(self.attn(pre_norm, context))) + x
        out = self.mlp(self.LN_mid(attn)) + attn

        return out


class SelfAttentionBlock(CrossAttentionBlock):
    """A.K.A. a Transformer pre-norm block"""
    def forward(self, x, *_):
        return super().forward(x)


def fourier_encode(x, max_freq, num_bands=4, base=2):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    # scales = torch.logspace(0., log(max_freq / 2) / log(base), num_bands, base=base, device=device, dtype=dtype)
    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * math.pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


def fourier_pos(batch_size, axis, max_freq, num_freq_bands, freq_base, device):
    # calculate fourier encoded positions in the range of [-1, 1], for all axis
    axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))
    pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
    enc_pos = fourier_encode(pos, max_freq, num_freq_bands, base=freq_base)
    enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
    enc_pos = repeat(enc_pos, '... -> b ...', b=batch_size)
    return enc_pos


class AttentionPool(nn.Module):
    """A.K.A. a Transformer"""
    def __init__(self, channels_in=32, heads=None, output_dim=None, depth=1, input_shape=None):
        super().__init__()

        self.input_shape = input_shape

        if input_shape is not None:
            channels_in = input_shape[-3]

        if output_dim is None:
            output_dim = channels_in

        if heads is None:
            heads = math.gcd(output_dim, 8)  # Approx 8

        self.pool = nn.Sequential(Utils.ChSwap,
                                  # "Transformer"
                                  *[SelfAttentionBlock(dim=channels_in, heads=heads) for _ in range(depth)],
                                  nn.Linear(channels_in, output_dim),
                                  Utils.ChSwap,
                                  AvgPool())

    def repr_shape(self, c, h, w):
        return Utils.cnn_feature_shape([c, h, w], self.pool)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        x = torch.cat(
            [context.view(*context.shape[:-3], -1, *self.input_shape[1:]) if len(context.shape) > 3
             else context.view(*context.shape[:-1], -1, *self.input_shape[1:]) if context.shape[-1]
                                                                                  % math.prod(self.input_shape[1:]) == 0
            else context.view(*context.shape, 1, 1).expand(*context.shape, *self.input_shape[1:])
             for context in x if context.nelement() > 0], dim=-3)
        # Conserve leading dims
        lead_shape = x.shape[:-3]
        # Operate on last 3 dims
        x = x.view(-1, *x.shape[-3:])

        x = self.pool(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out
