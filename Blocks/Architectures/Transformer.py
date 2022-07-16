import math

from einops import rearrange, repeat

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

        # Dimensions
        self.mlp_hidden_dim = mlp_hidden_dim or self.attend.value_dim * 4

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


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, G: int, M: int, F_dim: int, H_dim: int, D: int, gamma: float):
        """Borrowed from https://github.com/willGuimont/learnable_fourier_positional_encoding/
        Learnable Fourier Features from https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1)

        Implementation of Algorithm 1: Compute the Fourier feature positional encoding of a multi-dimensional position
        Computes the positional encoding of a tensor of shape [N, G, M]

        :param G: positional groups (positions in different groups are independent)
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr"""
        super().__init__()

        self.G, self.M, self.F_dim, self.H_dim, self.D, self.gamma \
            = map(torch.as_tensor, [G, M, F_dim, H_dim, D, gamma])

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)

        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G)
        )

        # Initialize weights
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [N, G, M] that represents N positions where each position is in the shape of [G, M],
                  where G is the positional group and each group has M-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        N, G, M = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / torch.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
        Y = self.mlp(F)
        # Step 3. Reshape to x's shape
        PEx = Y.reshape((N, self.D))
        return PEx


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
    For consistency with Vision models, assumes channels-first
    Generalized to arbitrary spatial dimensions"""
    def __init__(self, input_shape=(32,), num_heads=None, depth=1, channels_first=True):
        super().__init__()

        # Dimensions
        self.input_shape = input_shape

        positional_encodings = LearnableFourierPositionalEncoding()

        self.transformer = nn.Sequential(positional_encodings, *[SelfAttentionBlock(input_shape, num_heads,
                                                                                    channels_first=channels_first)
                                                                 for _ in range(depth)])

    def repr_shape(self, _):
        return _  # Isotropic, conserves dimensions

    def forward(self, *obs):
        # Shape broadcasting ...

        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        obs = torch.cat(
            [context.view(*context.shape[:-3], -1, *self.input_shape[1:]) if len(context.shape) > 3
             else context.view(*context.shape[:-1], -1, *self.input_shape[1:]) if context.shape[-1]
                                                                                  % math.prod(self.input_shape[1:]) == 0
            else context.view(*context.shape, 1, 1).expand(*context.shape, *self.input_shape[1:])
             for context in obs if context.nelement() > 0], dim=-3)
        # Conserve leading dims
        lead_shape = obs.shape[:-3]

        # Operate on last 3 dims
        obs = obs.view(-1, *obs.shape[-3:])

        # Encode & attend

        output = self.transformer(obs)

        # Restore leading dims
        return output.view(*lead_shape, *obs.shape[1:])
