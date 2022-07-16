import math

from einops import rearrange, repeat

import torch
from torch import nn

import Utils

from Blocks.Architectures.MLP import MLP
from Blocks.Architectures.MultiHeadAttention import CrossAttention
from Blocks.Architectures.Vision.CNN import AvgPool


class AttentionBlock(nn.Module):
    """
    A Transformer pre-norm block (https://arxiv.org/pdf/2002.04745.pdf)
    Generalized to cross-attend from inputs to contexts, broadcasting various shapes, with support for "talking heads"
    and ReLA". For consistency with Vision models, assumes channels-first!
    """
    def __init__(self, input_shape=(32,), num_heads=None, context_dim=None, query_key_dim=None, value_dim=None,
                 mlp_hidden_dim=None, dropout=0, talking_heads=False, rela=False, channels_first=True):
        super().__init__()

        self.LayerNormPre = nn.LayerNorm(self.attend.input_dim)

        # Multi-Head Dot-Product Attention (MHDPA) from inputs to context
        self.attend = CrossAttention(input_shape, num_heads, context_dim, query_key_dim, value_dim, talking_heads, rela,
                                     channels_first)

        # "Rectified-Linear Attention (ReLA)" (https://arxiv.org/abs/2104.07012)
        if rela:
            self.LayerNormReLA = nn.LayerNorm(self.attend.value_dim)

        if self.attend.num_heads > 1:
            self.map_heads = nn.Sequential(nn.Linear(self.attend.value_dim, self.attend.input_dim),
                                           nn.Dropout(dropout))

        self.LayerNormPost = nn.LayerNorm(self.attend.input_dim)

        # Dimensions
        self.mlp_hidden_dim = mlp_hidden_dim or self.attend.value_dim * 4

        self.MLP = nn.Sequential(MLP(self.attend.input_dim, self.attend.input_dim, self.mlp_hidden_dim,
                                     depth=1, non_linearity=nn.GELU(), dropout=dropout), nn.Dropout(dropout))

    def repr_shape(self, *_):
        # Conserves spatial dimensions, maps channel dim to value-dim
        return (self.attend.value_dim, *_[1:]) if self.channels_first \
            else (*_[:-1], self.attend.value_dim)

    def forward(self, input, context=None):
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

        return output


class CrossAttentionBlock(AttentionBlock):
    """Cross-Attention Block, same as the Attention Block"""


class SelfAttentionBlock(AttentionBlock):
    """A.K.A. a Transformer pre-norm block except input=context"""
    def forward(self, input, *_):
        return super().forward(input)


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


class Transformer(nn.Module):
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