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

        if channels_first:
            input_shape = list(reversed(input_shape))  # Assumes invariance to spatial dimensions

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
    def __init__(self, input_shape=(32,), fourier_dim=None, hidden_dim=None, output_dim=None, channels_first=True):
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

        fourier_dims = fourier_dim or self.input_dim
        self.fourier_dim = -(-fourier_dims // 2)  # Round up

        self.hidden_dim = hidden_dim or self.input_dim
        self.output_dim = output_dim or self.input_dim

        self.scale = 1 / math.sqrt(self.fourier_dim)

        # Projections
        self.Linear = nn.Linear(self.input_dim, self.fourier_dim, bias=False)
        self.MLP = MLP(self.fourier_dim * 2, self.output_dim, self.hidden_dim, 1, nn.GELU())

        # Initialize weights
        nn.init.normal_(self.Linear.weight.data)

    def repr_shape(self, *_):
        # Conserves spatial dimensions
        return (self.output_dim, *_[1:]) if self.channels_first \
            else (*_[:-1], self.output_dim)

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
        cosines *= self.scale
        sines *= self.scale

        # Fourier features via MLP
        fourier_features = self.MLP(cosines, sines)

        # Restores original dims
        output = fourier_features.view(*lead_dims, -1)

        return Utils.ChSwap(output, False) if self.channels_first \
            else output


# def fourier_pos(batch_size, axes, max_freq, num_freq_bands=4):
#     # Calculate fourier encoded positions in the range of [-1, 1], for all axes
#     axis_pos = list(map(lambda axis: torch.linspace(-1., 1., steps=axis, device=device), axes))
#     pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
#
#     x = pos.unsqueeze(-1)
#     device, dtype, orig_x = x.device, x.dtype, x
#
#     # scales = torch.logspace(0., log(max_freq / 2) / log(base), num_bands, base=base, device=device, dtype=dtype)
#     scales = torch.linspace(1., max_freq / 2, num_freq_bands, device=device, dtype=dtype)
#     scales = scales[(*((None,) * (len(x.shape) - 1)), ...)]
#
#     x = x * scales * math.pi
#     x = torch.cat([x.sin(), x.cos()], dim=-1)
#
#     x = torch.cat((x, orig_x), dim=-1)
#
#     enc_pos = x.flatten(-2).expand(batch_size, *x.shape[1:])
#     return enc_pos
#
#
# class FourierPositionalEncodings(nn.Module):
#     def __init__(self, input_shape=(32,)):
#         super().__init__()
#
#         channels = int(np.ceil(channels / 4) * 2)
#
#         inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
#
#         self.cached_penc = None
#         batch_size, x, y, orig_ch = input.shape
#
#         # Ranges
#         pos_x = torch.arange(x, device=input.device).type(self.inv_freq.type())
#         pos_y = torch.arange(y, device=input.device).type(self.inv_freq.type())
#
#         # Grid (ranges x freq)
#         sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
#         sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
#
#         # Sin and cos concat
#         emb_x = torch.cat([sin_inp_x.sin(), sin_inp_x.cos()], -1).unsqueeze(1)
#
#         # Sin and cos concat
#         emb_y = torch.cat([sin_inp_y.sin(), sin_inp_y.cos()], -1)
#
#         # Concat
#         emb = torch.cat([emb_x, emb_y], -1)
#
#     def forward(self, input):
#         # Apply per batch
#         pass


class PositionalEncodings(nn.Module):
    """
    Sine-cosine positional encodings
    Generalized to adapt to arbitrary spatial dimensions. For consistency with Vision models,
    assumes channels-first! Automatically adds when encoding size is same as input, otherwise concatenates.
    """
    def __init__(self, input_shape=(7, 32, 32), dropout=0.1, size=None, max_spatial_lens=None, channels_first=True):
        super().__init__()

        # Dimensions

        self.channels_first = channels_first

        self.input_dim = input_shape[0] if channels_first else input_shape[-1]

        # Max spatial lengths (for variable sequences)
        if max_spatial_lens is None:
            max_spatial_lens = input_shape[1:] if channels_first else input_shape[:-1]

        self.size = max(size or self.input_dim, len(max_spatial_lens))

        div_term = torch.exp(torch.arange(0, self.size, 2) * (-math.log(10000.0) / self.size))

        positions = map(torch.arange, max_spatial_lens)

        positional_encodings = torch.zeros(*max_spatial_lens, self.size)

        for i, position in enumerate(positions):
            positional_encodings[..., 0::len(max_spatial_lens)] = torch.sin(position.unsqueeze(1) * div_term)
            positional_encodings[..., i::len(max_spatial_lens)] = torch.cos(position.unsqueeze(1) * div_term)

        self.register_buffer('positional_encodings', positional_encodings)

        self.dropout = nn.Dropout(p=dropout)

    def repr_shape(self, *_):
        # Conserves shape when additive (if spatial axes ≤ input dim), else concatenates
        return _ if self.input_dim == self.size \
            else (self.input_dim + self.size, *_[1:]) if self.channels_first else (*_[:-1], self.input_dim + self.size)

    def forward(self, input):
        # Permute as channels-last
        if self.channels_first:
            input = Utils.ChSwap(input, False)

        positions = self.positional_encodings[list(map(slice, input.shape[1:-1]))]

        # Add or concatenate
        encodings = self.dropout(input + positions) if self.input_dim == self.size \
            else torch.cat([input, positions.expand(input.shape[0], *positions.shape)], -1)

        # To channels-first if needed
        return Utils.ChSwap(encodings, False) if self.channels_first \
            else encodings


class Transformer(nn.Module):
    """A Transformer
    For consistency with Vision models, assumes channels-first!
    Generalized to arbitrary spatial dimensions"""
    def __init__(self, input_shape=(32,), num_heads=None, depth=1, channels_first=True,
                 learnable_positional_encodings=False, positional_encodings=True):
        super().__init__()

        positional_encodings = LearnableFourierPositionalEncodings if learnable_positional_encodings \
            else PositionalEncodings if positional_encodings else nn.Identity()

        positional_encodings = positional_encodings(input_shape, channels_first=channels_first)

        self.shape = Utils.cnn_feature_shape(input_shape, positional_encodings)

        self.transformer = nn.Sequential(positional_encodings, *[SelfAttentionBlock(self.shape, num_heads,
                                                                                    channels_first=channels_first)
                                                                 for _ in range(depth)])

    def repr_shape(self, *_):
        return self.shape

    def forward(self, obs):
        return self.transformer(obs)
