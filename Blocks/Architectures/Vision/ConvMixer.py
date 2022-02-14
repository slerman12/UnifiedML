# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch.nn as nn

from Blocks.Architectures.Residual import Residual


class ConvMixer(nn.Module):
    def __init__(self, input_shape, dim=32, depth=3, kernel_size=9, patch_size=7, n_classes=1000):
        super().__init__()
        in_channels = input_shape[0]

        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)],
            nn.AdaptiveAvgPool2d((1,1)),
            # nn.Flatten(),
            # nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        return self.CNN(x)
