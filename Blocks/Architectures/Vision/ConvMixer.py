# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

import Utils
from Blocks.Architectures.Residual import Residual


class ConvMixer(nn.Module):
    def __init__(self, input_shape, dim=32, depth=3, kernel_size=9, patch_size=7, output_dim=None):
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
            nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                          nn.Flatten(),
                          nn.Linear(dim, output_dim)) if output_dim is not None
            else nn.Identity()
        )

    def feature_shape(self, h, w):
        return Utils.cnn_feature_shape(h, w, self.CNN)

    def forward(self, *x):
        x = list(x)
        x[0] = x[0].view(-1, *self.input_shape)

        # Optionally append context to channels assuming dimensions allow
        if len(x) > 1:
            x[1:] = [context.reshape(x[0].shape[0], context.shape[-1], 1, 1).expand(-1, -1, *self.input_shape[1:])
                     for context in x[1:]]

        x = torch.cat(x, 1)

        return self.CNN(x)
