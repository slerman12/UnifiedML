# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
from torch import nn

import Utils

from Blocks.Architectures.Residual import Residual


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super().__init__()

        if down_sample is None and (in_channels != out_channels or stride != 1):
            down_sample = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                  kernel_size=1, stride=stride, bias=False),
                                        nn.BatchNorm2d(out_channels))

        pre_residual = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                               kernel_size=3, padding=1, stride=stride, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels, out_channels,
                                               kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels))

        self.Residual_block = nn.Sequential(Residual(pre_residual, down_sample),
                                            nn.ReLU(inplace=True))

    def feature_shape(self, h, w):
        return Utils.cnn_feature_shape(h, w, self.Residual_block)

    def forward(self, x):
        return self.Residual_block(x)


class MiniResNet(nn.Module):
    def __init__(self, input_shape, dims=None, depths=None, output_dim=None):
        super().__init__()

        self.input_shape = input_shape
        in_channels = input_shape[0]

        if dims is None:
            dims = [32, 32]

        if depths is None:
            depths = [3]

        # CNN ResNet-ish
        self.CNN = nn.Sequential(nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1, bias=False),
                                 nn.BatchNorm2d(dims[0]),
                                 nn.ReLU(inplace=True),
                                 *[ResidualBlock(dims[i + (j > 0)], dims[i + 1], 1 + (i > 0 and j > 0))
                                   for i, depth in enumerate(depths)
                                   for j in range(depth)])

        self.projection = nn.Identity() if output_dim is None \
            else nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                               nn.Flatten(),
                               nn.Linear(dims[-1], 1024),
                               nn.ReLU(inplace=True),
                               nn.Linear(1024, output_dim))

    def feature_shape(self, h, w):
        return Utils.cnn_feature_shape(h, w, self.CNN)

    def forward(self, *x):
        # Optionally append context to channels assuming dimensions allow
        if len(x) > 1:
            # Warning: merely reshapes context where permitted, rather than expanding it to height and width
            x = [context.view(*context.shape[:-1], -1, *self.input_shape[1:]) if context.shape[-1]
                                                                                 % math.prod(self.input_shape) == 0
                 else context.view(*context.shape[:-1], -1, 1, 1).expand(*context.shape[:-1], -1, *self.input_shape[1:])
                 for context in x if len(context.shape) < 4 and context.shape[-1]]
        x = torch.cat(x, -3)

        # Conserve leading dims
        lead_shape = x.shape[:-3]

        # Operate on last 3 dims
        x = x.view(-1, *self.input_shape)

        out = self.CNN(x)

        out = self.projection(out)

        # Restore leading dims
        out = out.view(*lead_shape, *out.shape[1:])

        return out
