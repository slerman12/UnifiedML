# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

from einops import rearrange

import torch
from torch import nn

import Utils


class Attention(nn.Module):
    """
    Multi-head dot-product attention (MHDPA) from inputs to contexts (Cross-Attention)
    (https://arxiv.org/abs/1706.03762?context=cs)

    All you need

    Generalized to arbitrary input shapes, and includes options for "talking heads" and "ReLA".
    For consistency with Vision models, defaults to channels-first!
    """

    def __init__(self, input_shape=(32,), num_heads=None, context_dim=None, query_key_dim=None, value_dim=None,
                 talking_heads=False, rela=False, channels_first=True):
        super().__init__()

        self.channels_first = channels_first

        # Dimensions
        self.input_dim = input_shape[0] if channels_first \
            else input_shape[-1]

        # Defaults
        self.context_dim = context_dim or self.input_dim
        self.query_key_dim = query_key_dim or self.input_dim
        self.value_dim = value_dim or self.input_dim

        self.num_heads = num_heads or math.gcd(8, value_dim)

        assert self.value_dim % self.num_heads == 0, \
            f'Value dim={self.value_dim} must be divisible by heads={self.num_heads}'

        # Linear QKV-projections  TODO pass in and call Utils.instantiate - may not need projection, e.g. Identity
        self.to_query = nn.Linear(self.input_dim, self.query_key_dim, bias=False)
        self.to_key_value = nn.Linear(self.context_dim, self.query_key_dim + self.value_dim, bias=False)

        # Can access attention weights
        self.saved_attention_weights = None

        # Additional options

        # "Talking heads" (https://arxiv.org/abs/2003.02436)
        if talking_heads:
            self.talk_h = nn.Sequential(Utils.ChSwap, nn.Linear(self.num_heads, self.num_heads, bias=False),
                                        nn.LayerNorm(self.num_heads), Utils.ChSwap)

        # "Rectified-Linear Attention (ReLA)" (https://arxiv.org/abs/2104.07012)
        if rela:
            self.rela = nn.ReLU(inplace=True)

    def repr_shape(self, *_):
        # Conserves spatial dimensions, maps channel dim to value-dim
        return (self.value_dim, *_[1:]) if self.channels_first \
            else (*_[:-1], self.value_dim)

    def forward(self, input, context=None):
        if context is None:
            context = input  # Self-attention

        """
        Adapt to:
        1. no batch dim, no spatial dim, only channel dim
        2. no batch dim, spatial dim, channel dim
        3. batch dim, no spatial dim, channel dim
        4. batch dim, spatial dims, channel dim
        Can assume at least context or input has batch dim
        If both 2d, assume both b x c
        """

        if len(input.shape) == len(context.shape) == 2:
            input_axes = context_axes = 'b'  # Output: b x d
        elif len(input.shape) == 1:
            input_axes, context_axes = '', 'b' if len(context.shape) == 2 else 'b j'  # Output: b x d
        elif len(context.shape) == 1:
            input_axes, context_axes = 'b' if len(input.shape) == 2 else 'b i', ''  # Output: b x d or b x i x d
        elif len(input.shape) == 2:
            input_axes, context_axes = 'b' if input.shape[0] == context.shape[0] else 'n', 'b j'  # Output: b x d
        elif len(context.shape) == 2:
            input_axes, context_axes = 'b i', 'b ' if input.shape[0] == context.shape[0] else 'j'  # Output: b x i x d
        else:
            input_axes, context_axes = 'b i', 'b j'  # Output: b x i x d, standard case

        input_has_batch_and_spatial_dims = input_axes == 'b i'
        context_has_batch_and_spatial_dims = context_axes == 'b j'

        # Permute as channels-last
        if self.channels_first:
            input, context = map(Utils.ChSwap, [input, context])

        # Preserve leading dims
        lead_dims = input.shape[:-1]

        # Flatten intermediary spatial dims
        if input_has_batch_and_spatial_dims:
            input = input.flatten(1, -2)
        if context_has_batch_and_spatial_dims:
            context = context.flatten(1, -2)

        # Validate shapes
        assert input.shape[-1] == self.input_dim, f'Unexpected input shape {input.shape[-1]}≠{self.input_dim}'
        assert context.shape[-1] == self.context_dim, f'Unexpected context shape {context.shape[-1]}≠{self.context_dim}'

        query = self.to_query(input)
        key, value = self.to_key_value(context).tensor_split((self.query_key_dim,), -1)  # Split into KV

        get_pattern = lambda axes, dim: dim if dim in axes else ''
        input_b, context_b = get_pattern(input_axes, 'b'), get_pattern(context_axes, 'b')
        i, j = get_pattern(input_axes, 'i'), get_pattern(context_axes, 'j')

        # Heads-first  TODO input/context may not need heads separated
        query = rearrange(query, f'{input_axes} (h c) -> {input_b} h {i} c', h=self.num_heads)
        key, value = [rearrange(proj, f'{context_axes} (h c) -> {context_b} h {j} c', h=self.num_heads)
                      for proj in (key, value)]

        # Scale (Q / sqrt(d))
        query *= query.shape[-1] ** -0.5

        # Multiply (W = Q * K)
        self.saved_attention_weights = torch.einsum(f'{input_b} h {i} c, {context_b} h {j} c -> b h {i} {j}',
                                                    query, key)

        # Normalize
        self.saved_attention_weights \
            = self.saved_attention_weights - self.saved_attention_weights.amax(dim=-1, keepdim=True).detach()

        # Softmax
        attention_weights = self.rela(self.saved_attention_weights) if hasattr(self, 'rela') \
            else self.saved_attention_weights.softmax(dim=-1)

        # "Talking heads"
        if hasattr(self, 'talk_h'):
            attention_weights = self.talk_h(attention_weights)

        # Attend (W * V)
        attention = torch.matmul(attention_weights, value)

        # Heads-last-concatenated
        output = rearrange(attention, 'b h n d -> b n (h d)')

        # Restores original leading dims
        output = output.view(*lead_dims, -1)

        # Convert to channels-first if needed
        return Utils.ChSwap(output) if self.channels_first \
            else output


class ReLA(Attention):
    """ReLA: Rectified linear attention (https://arxiv.org/abs/2104.07012)"""

    def __init__(self, dim=32, num_heads=None, context_dim=None, query_key_dim=None, value_dim=None):
        super().__init__(dim, num_heads, context_dim, query_key_dim, value_dim, False, True)


class CrossAttention(Attention):
    """Cross-attention, same as Attention"""


class SelfAttention(Attention):
    """Self-attention, just cross-attention except context = input"""

    def forward(self, input, *_):
        return super().forward(input)
