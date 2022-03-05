# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
from functools import partial
import copy

from einops import rearrange
from opt_einsum_torch import EinsumPlanner

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from Blocks.Architectures.MLP import MLP

import Utils


class CrossAttention(nn.Module):
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

        scale = q.shape[-1] ** -0.5
        q = q * scale

        pattern = 'h i d, b h j d -> b h i j' if tokens else 'b h i d, b h j d -> b h i j'

        # Memory efficient toggle
        mem_efficient = False
        if mem_efficient:
            attn, weights = mem_efficient_attend(q, k, v, pattern=pattern)
        else:
            self.dots = torch.einsum(pattern, q, k)

            if self.relu is None:
                # self.dots = self.dots - self.dots.amax(dim=-1, keepdim=True).detach()
                weights = self.dots.softmax(dim=-1)
            else:
                weights = self.relu(self.dots)

            if 0 < mem_limit < 1:
                weights = weights.to(q.device)

            # "Talking heads"
            weights = self.talk_h(weights)

            attn = einsum('b h i j, b h j d -> b h i d', weights, v)

        out = rearrange(attn, 'b h n d -> b n (h d)')

        # Restores original shape
        if not tokens:
            out = out.view(*shape[:-1], -1)

        if 0 < mem_limit < 1:
            out = out.to(q.device)

        return out


class ReLA(CrossAttention):
    """ReLA: Rectified linear attention"""
    def __init__(self, dim=32, heads=None, context_dim=None, value_dim=None):
        super().__init__(dim, heads, context_dim, value_dim, False, True)


# Memory-efficient attention https://arxiv.org/abs/2112.05682
# https://github.com/lucidrains/memory-efficient-attention-pytorch
def mem_efficient_attend(q, k, v, q_bucket_size=512, k_bucket_size=1024, eps=1e-8,
                         pattern='b h i d, b h j d -> b h i j'):
    def chunk(q, k, v):
        weight = torch.einsum(pattern, q, k)

        weight_max = weight.amax(dim=-1, keepdim=True).detach()
        weight = weight - weight_max

        exp_weight = weight.exp()
        weighted_value = torch.einsum('b h i j, b h j d -> b h i d', exp_weight, v)

        return exp_weight.sum(dim=-1), weighted_value, rearrange(weight_max, '... 1 -> ...')

    chunk = partial(checkpoint, chunk)

    # Chunk all the inputs

    q_chunks = q.split(q_bucket_size, dim=-2)
    k_chunks = k.split(k_bucket_size, dim=-2)
    v_chunks = v.split(k_bucket_size, dim=-2)

    # Loop through all chunks and accumulate

    out = []
    weights = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = []
        weighted_values = []
        weight_maxes = []

        for k_index, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):

            exp_weight_chunk, weighted_value_chunk, weight_max_chunk = \
                chunk(q_chunk, k_chunk, v_chunk)

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        weight_maxes = torch.stack(weight_maxes, dim=-1)

        weighted_values = torch.stack(weighted_values, dim=-1)
        exp_weights = torch.stack(exp_weights, dim=-1)

        global_max = weight_maxes.amax(dim=-1, keepdim=True)
        renorm_factor = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm_factor
        weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

        all_values = weighted_values.sum(dim=-1)
        all_weights = exp_weights.sum(dim=-1)

        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        out.append(normalized_values)
        weights.append(exp_weights)

    out = torch.cat(out, dim=-2)
    weights = torch.cat(weights, dim=-3)

    return out, weights


# A minimalist implementation using only Pytorch natives
class CrossAttend(nn.Module):
    def __init__(self, dim=32, heads=8, context_dim=None, value_dim=None, *_):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, heads, kdim=context_dim, vdim=value_dim, batch_first=True)

    def forward(self, x, context):
        # Conserves shape
        mid_shape = x.shape[1:-1]

        x = x.flatten(1, -2)
        context = context.flatten(1, -2)

        attn, self.dots = self.attn(x, context, context)

        # Restores original shape
        return attn.view(-1, *mid_shape, attn.shape[-1])


class SelfAttention(CrossAttention):
    def forward(self, x, *_):
        return super().forward(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=32, heads=1, context_dim=None, value_dim=None, talk_h=False, relu=False,
                 lr=None, weight_decay=0, ema_tau=None):
        super().__init__()

        context_dim = dim if context_dim is None \
            else context_dim

        value_dim = dim if value_dim is None \
            else value_dim

        self.heads = math.gcd(8, value_dim) if heads is None \
            else heads

        self.value_dim = value_dim

        self.attn = CrossAttention(dim, self.heads, context_dim, value_dim, talk_h, relu)
        self.mlp = MLP(value_dim, value_dim, value_dim, 1, nn.GELU())

        self.ln_attn = nn.LayerNorm(value_dim)
        self.ln = nn.LayerNorm(value_dim)

        self.init(lr, weight_decay, ema_tau)

    def repr_shape(self, c, h, w):
        return self.value_dim, h, w  # Assumes channels last

    def init(self, lr=None, weight_decay=0, ema_tau=None):
        # Initialize weights
        self.apply(Utils.weight_init)

        # Optimizer
        if lr is not None:
            self.optim = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # EMA
        if ema_tau is not None:
            self.ema = copy.deepcopy(self)
            self.ema_tau = ema_tau

    def forward(self, x, context=None):
        if context is None:
            context = x

        attn = self.ln_attn(self.attn(x, context)) + x
        out = self.ln(self.mlp(attn)) + attn

        return out


class SelfAttentionBlock(CrossAttentionBlock):
    def forward(self, x, *_):
        return super().forward(x)


class AttentionPool(nn.Module):
    def __init__(self, channels_in=32, heads=None, output_dim=None, depth=1, recursions=0, input_shape=None):
        super().__init__()

        self.input_shape = input_shape

        if input_shape is not None:
            channels_in = input_shape[-3]

        if output_dim is None:
            output_dim = channels_in

        if heads is None:
            heads = math.gcd(output_dim, 8)  # Approx 8

        self.pool = nn.Sequential(Utils.ChSwap,
                                  # Alternatively could also recurse
                                  *([SelfAttentionBlock(channels_in, heads)] * recursions),
                                  # "Transformer"
                                  *[SelfAttentionBlock(dim=channels_in if i == 0 else output_dim, heads=heads,
                                                       value_dim=output_dim) for i in range(depth)],
                                  Utils.ChSwap,
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Flatten(-3))

    def repr_shape(self, c, h, w):
        return Utils.cnn_feature_shape(c, h, w, self.pool)

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
