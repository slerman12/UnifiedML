# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

import Utils


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # Depth-wise conv
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim),
                                 nn.GELU(),
                                 nn.Linear(4 * dim, dim))
        self.gamma = nn.Parameter(torch.full((dim,), 1e-6))

    def feature_shape(self, h, w):
        return Utils.cnn_feature_shape(h, w, self.conv)

    def forward(self, x):
        input = x
        x = self.conv(x)
        x = x.transpose(-1, -3)  # Channel swap
        x = self.ln(x)
        x = self.mlp(x)
        x = self.gamma * x
        x = x.transpose(-1, -3)  # Channel swap
        return x + input


class ConvNeXt(nn.Module):
    r""" ConvNeXt  `A ConvNet for the 2020s` https://arxiv.org/pdf/2201.03545.pdf"""
    def __init__(self, input_shape, dims=None, depths=None, output_dim=None):
        super().__init__()

        channels_in = input_shape[0]

        if dims is None:
            # dims = [channels_in, 96, 192, 384, 768]  # TinyConvNeXt
            dims = [channels_in, 96, 192, 32]

        if depths is None:
            # depths = [3, 3, 9, 3]  # TinyConvNeXt
            depths = [1, 1, 3]

        self.CNN = nn.Sequential(*[nn.Sequential(nn.Conv2d(dims[i],
                                                           dims[i + 1],
                                                           kernel_size=4 if i == 0 else 2,
                                                           stride=4 if i == 0 else 2),  # Conv
                                                 nn.Sequential(Utils.ChannelSwap(),
                                                               nn.LayerNorm(dims[i + 1]),
                                                               Utils.ChannelSwap()) if i < 3
                                                 else nn.Identity(),  # LayerNorm
                                                 *[ConvNeXtBlock(dims[i + 1])
                                                   for _ in range(depth)])  # Conv, MLP, Residuals
                                   for i, depth in enumerate(depths)],
                                 nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                               nn.Sequential(Utils.ChannelSwap(),
                                                             nn.LayerNorm(dims[-1]),
                                                             Utils.ChannelSwap()),  # LayerNorm
                                               nn.Flatten(),
                                               nn.Linear(dims[-1], output_dim)) if output_dim is not None
                                 else nn.Identity())

        def weight_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

        self.apply(weight_init)

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
