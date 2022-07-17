# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

import Utils

from Blocks.Architectures.MLP import MLP
from Blocks.Architectures.MultiHeadAttention import CrossAttention


class AttentionBlock(nn.Module):
    """
    A Transformer pre-norm block (https://arxiv.org/pdf/2002.04745.pdf)
    Generalized to cross-attend from inputs to contexts, broadcasting various shapes, with support for "talking heads"
    and "ReLA". For consistency with Vision models, assumes channels-first!
    """
    def __init__(self, input_shape=(32,), num_heads=None, context_dim=None, query_key_dim=None, value_dim=None,
                 mlp_hidden_dim=None, dropout=0, talking_heads=False, rela=False, channels_first=True):
        super().__init__()

        self.channels_first = channels_first

        # Multi-Head Dot-Product Attention (MHDPA) from inputs to context
        self.attend = CrossAttention(input_shape, num_heads, context_dim, query_key_dim, value_dim, talking_heads, rela,
                                     channels_first=False)

        self.LayerNormPre = nn.LayerNorm(self.attend.input_dim)  # Applied before the above attention

        # "Rectified-Linear Attention (ReLA)" (https://arxiv.org/abs/2104.07012)
        if rela:
            self.LayerNormReLA = nn.LayerNorm(self.attend.value_dim)

        if self.attend.num_heads > 1:
            self.map_heads = nn.Sequential(nn.Linear(self.attend.value_dim, self.attend.input_dim),
                                           nn.Dropout(dropout))

        self.LayerNormPost = nn.LayerNorm(self.attend.input_dim)

        self.mlp_hidden_dim = mlp_hidden_dim or self.attend.value_dim * 4  # MLP dimension

        self.MLP = nn.Sequential(MLP(self.attend.input_dim, self.attend.input_dim, self.mlp_hidden_dim,
                                     depth=1, non_linearity=nn.GELU(), dropout=dropout), nn.Dropout(dropout))

    def repr_shape(self, *_):
        # Isotropic, conserves dimensions
        return _

    def forward(self, input, context=None):
        # To channels-last
        if self.channels_first:
            input = Utils.ChSwap(input, False)

            if context is not None:
                context = Utils.ChSwap(context, False)

        pre_norm = self.LayerNormPre(input)

        if context is None:
            context = pre_norm

        attention = self.attend(pre_norm, context)

        if hasattr(self, 'LayerNormReLA'):
            attention = self.LayerNormReLA(attention)

        if hasattr(self, 'map_heads'):
            attention = self.map_heads(attention)

        residual = attention + input
        output = self.MLP(self.LayerNormPost(residual)) + residual

        return Utils.ChSwap(output, False) if self.channels_first \
            else output


class CrossAttentionBlock(AttentionBlock):
    """Cross-Attention Block, same as the Attention Block"""


class SelfAttentionBlock(AttentionBlock):
    """A.K.A. a Transformer pre-norm block except input=context"""
    def forward(self, input, *_):
        return super().forward(input)


class LearnableFourierPositionalEncodings(nn.Module):
    def __init__(self, input_shape=(32,), fourier_dim=8, hidden_dim=8, output_dim=8, channels_first=True):
        """
        Learnable Fourier Features (https://arxiv.org/pdf/2106.02795.pdf)
        Generalized to adapt to arbitrary spatial dimensions. For consistency with Vision models,
        assumes channels-first!
        """
        super().__init__()

        # Dimensions

        self.channels_first = channels_first

        self.input_dim = input_shape if isinstance(input_shape, int) \
            else input_shape[0] if channels_first else input_shape[-1]

        self.scale = 1 / math.sqrt(fourier_dim)

        # Projections
        self.Linear = nn.Linear(self.input_dim, fourier_dim // 2, bias=False)
        self.MLP = MLP(fourier_dim, output_dim, hidden_dim, 1, nn.GELU())

        # Initialize weights
        nn.init.normal_(self.Linear.weight.data)

    def forward(self, input):
        # Permute as channels-last
        if self.channels_first:
            input = Utils.ChSwap(input, False)

        # Preserve batch/spatial dims
        lead_dims = input.shape[:-1]

        # Flatten intermediary spatial dims
        input = input.flatten(1, -2)

        # Linear-project features
        features = self.Linear(input)

        cosines, sines = torch.cos(features), torch.sin(features)

        # Fourier features
        fourier_features = self.scale * torch.cat([cosines, sines], dim=-1)

        # MLP
        output = self.MLP(fourier_features)

        # Restores original leading dims
        output = output.view(*lead_dims, -1)

        return Utils.ChSwap(output, False) if self.channels_first \
            else output


# def fourier_encode(x, max_freq, num_bands=4, base=2):
#     x = x.unsqueeze(-1)
#     device, dtype, orig_x = x.device, x.dtype, x
#
#     # scales = torch.logspace(0., log(max_freq / 2) / log(base), num_bands, base=base, device=device, dtype=dtype)
#     scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
#     scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]
#
#     x = x * scales * math.pi
#     x = torch.cat([x.sin(), x.cos()], dim=-1)
#     x = torch.cat((x, orig_x), dim=-1)
#     return x
#
#
# def fourier_pos(batch_size, axis, max_freq, num_freq_bands, freq_base, device):
#     # calculate fourier encoded positions in the range of [-1, 1], for all axis
#     axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))
#     pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
#     enc_pos = fourier_encode(pos, max_freq, num_freq_bands, base=freq_base)
#     enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
#     enc_pos = repeat(enc_pos, '... -> b ...', b=batch_size)
#     return enc_pos


class Transformer(nn.Module):
    """A Transformer
    For consistency with Vision models, assumes channels-first!
    Generalized to arbitrary spatial dimensions"""
    def __init__(self, input_shape=(32,), num_heads=None, depth=1, channels_first=True):
        super().__init__()

        # Dimensions
        self.input_shape = input_shape

        positional_encodings = LearnableFourierPositionalEncodings(channels_first=channels_first)

        self.transformer = nn.Sequential(positional_encodings, *[SelfAttentionBlock(input_shape, num_heads,
                                                                                    channels_first=channels_first)
                                                                 for _ in range(depth)])

    def repr_shape(self, _):
        return _  # Isotropic, conserves dimensions

    def forward(self, obs):
        return self.transformer(obs)
