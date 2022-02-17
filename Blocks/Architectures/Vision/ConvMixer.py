# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch
import torch.nn as nn

import Utils

from Blocks.Architectures.Residual import Residual


class ConvMixer(nn.Module):
    def __init__(self, input_shape, out_channels=32, depth=3, kernel_size=9, patch_size=7, output_dim=None):
        super().__init__()

        self.input_shape = input_shape
        in_channels = input_shape[0]

        self.trunk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size),
                                   nn.GELU(),
                                   nn.BatchNorm2d(out_channels))

        self.ConvMixer = nn.Sequential(*[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size, groups=out_channels, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(out_channels)
                )),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(out_channels)
            ) for _ in range(depth)]
        )

        self.projection = nn.Identity() if output_dim is None \
            else nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                               nn.Flatten(),
                               nn.Linear(out_channels, 1024),
                               nn.ReLU(inplace=True),
                               nn.Linear(1024, output_dim))

    def feature_shape(self, h, w):
        return Utils.cnn_feature_shape(h, w, self.ConvMixer)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        x = torch.cat(
            [context.view(*context.shape[:-3], -1, *self.input_shape[1:]) if len(context.shape) > 3
             else context.view(*context.shape[:-1], -1, *self.input_shape[1:]) if math.prod(self.input_shape[1:])
                                                                                  % context.shape[-1] == 0
             else context.view(*context.shape, 1, 1).expand(*context.shape, *self.input_shape[1:])
             for context in x if context.nelement() > 0], dim=-3)
        # Conserve leading dims
        lead_shape = x.shape[:-3]
        # Operate on last 3 dims
        x = x.view(-1, *x.shape[-3:])

        x = self.trunk(x)
        x = self.ConvMixer(x)
        x = self.projection(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out
