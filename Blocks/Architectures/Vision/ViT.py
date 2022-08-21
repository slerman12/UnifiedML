# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

import Utils

from Blocks.Architectures.Transformer import SelfAttentionBlock, LearnableFourierPositionalEncodings
from Blocks.Architectures.Vision.CNN import AvgPool


class ViT(nn.Module):
    """A Vision Transformer (https://arxiv.org/abs/2010.11929)
    Generalized to adapt to arbitrary temporal-spatial dimensions, assumes channels-first"""
    def __init__(self, input_shape=(32,), out_channels=32, patch_size=4, num_heads=None, depth=3, emb_dropout=0.1,
                 query_key_dim=None, mlp_hidden_dim=None, dropout=0.1, pool_type='cls', output_dim=None, fourier=False):
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = [input_shape]

        in_channels, *self.spatial_shape = input_shape

        # Assumes image/input divisible by patch size per each dim
        patches = nn.Conv2d(in_channels, out_channels, patch_size, stride=patch_size)
        token = CLSToken(Utils.cnn_feature_shape(input_shape, patches))
        shape = Utils.cnn_feature_shape(input_shape, token)
        positional_encodings = (LearnableFourierPositionalEncodings if fourier else LearnablePositionalEncodings)(shape)

        self.ViT = nn.Sequential(patches,
                                 token,
                                 positional_encodings,
                                 nn.Dropout(emb_dropout),
                                 *[SelfAttentionBlock(shape, num_heads, None, query_key_dim, mlp_hidden_dim, dropout)
                                   for _ in range(depth)],
                                 # Can CLS-pool and project to a specified output dim, optional
                                 nn.Identity() if output_dim is None else nn.Sequential(CLSPool() if pool_type == 'cls'
                                                                                        else AvgPool(),
                                                                                        nn.Linear(out_channels,
                                                                                                  output_dim)))

    def repr_shape(self, *_):
        return Utils.cnn_feature_shape(_, self.ViT, self.pool)

    def forward(self, *x):
        # Broadcasting

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

        # ViT

        x = self.ViT(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out


class LearnablePositionalEncodings(nn.Module):
    def __init__(self, input_shape=(32, 7, 7)):
        """
        Learnable positional encodings
        Generalized to adapt to arbitrary dimensions. Assumes channels-first!
        """
        super().__init__()

        # Dimensions

        self.in_channels, *self.spatial_shape = input_shape

        self.encoding = nn.Parameter(torch.randn(1, math.prod(self.spatial_shape) + 1, self.in_channels))

    def repr_shape(self, *_):
        return _  # Conserves shape

    def forward(self, x):
        # Collapse spatial dims
        x = x.flatten(2)

        x += self.encoding[:, :math.prod(self.spatial_shape) + 1]  # Add positional encodings

        # Restore spatial dims
        return x.view(*x.shape[:2], *self.spatial_shape)


class CLSToken(nn.Module):
    """Appends a CLS token, assumes channels-first  (https://arxiv.org/pdf/1810.04805.pdf section 3)"""
    def __init__(self, input_shape=(32,)):
        super().__init__()

        in_channels = input_shape if isinstance(input_shape, int) else input_shape[0]

        self.token = nn.Parameter(torch.randn(1, 1, in_channels))

    def repr_shape(self, c, h, *_):
        return c, h + 1, *_

    def forward(self, obs):
        return torch.cat([obs, self.token.expand_as(obs)], dim=1)


class CLSPool(nn.Module):
    """Selects the CLS token as the representative embedding, assuming channels-first"""
    def __init__(self, **_):
        super().__init__()

    def repr_shape(self, c, *_):
        return c, *(1,) * len(_)

    def forward(self, x):
        return x.flatten(2)[..., -1]

