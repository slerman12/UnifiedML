# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from torch import nn

import Utils

from Blocks.Architectures.Residual import Residual


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super().__init__()

        pre_residual = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                               kernel_size=3, padding=1, stride=stride, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(out_channels, out_channels,
                                               kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels))

        self.Residual_block = nn.Sequential(Residual(pre_residual, down_sample),
                                            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.Residual_block(x)

    def output_shape(self, height, width):
        return Utils.cnn_output_shape(height, width, self.Residual_block)


class MiniResNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_blocks, pre_residual=False):
        super().__init__()

        pre = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                            nn.BatchNorm2d(hidden_channels))

        # Add a concurrent stream to pre
        if pre_residual:
            pre = Residual(pre, down_sample=nn.Sequential(nn.Conv2d(in_channels, hidden_channels,
                                                                    kernel_size=3, padding=1),
                                                          nn.BatchNorm2d(hidden_channels)))

        # CNN ResNet-ish
        self.CNN = nn.Sequential(pre,
                                 nn.ReLU(inplace=True),  # MaxPool after this?
                                 *[ResidualBlock(hidden_channels, hidden_channels)
                                   for _ in range(num_blocks)],
                                 nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        return self.CNN(x)
