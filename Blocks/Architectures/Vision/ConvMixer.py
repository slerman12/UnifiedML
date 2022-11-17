# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch.nn as nn

from Blocks.Architectures.Residual import Residual
from Blocks.Architectures.Vision.CNN import AvgPool, broadcast

import Utils


class ConvMixer(nn.Module):
    def __init__(self, input_shape, out_channels=32, depth=3, kernel_size=9, patch_size=7, output_shape=None):
        super().__init__()

        self.input_shape = in_channels, *_ = input_shape

        output_dim = Utils.prod(output_shape)

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
        ) for _ in range(depth)])

        self.project = nn.Identity() if output_dim is None \
            else nn.Sequential(AvgPool(), nn.Linear(out_channels, output_dim))

    def repr_shape(self, *_):
        return Utils.cnn_feature_shape(_, self.trunk, self.ConvMixer, self.project)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = broadcast(self.input_shape, x)

        x = self.trunk(x)
        x = self.ConvMixer(x)
        x = self.project(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out
