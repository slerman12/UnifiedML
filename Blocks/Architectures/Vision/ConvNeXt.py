# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

from Blocks.Architectures.Vision.CNN import AvgPool, broadcast

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

    def repr_shape(self, *_):
        return Utils.cnn_feature_shape(_, self.conv)

    def forward(self, x):
        input = x
        x = self.conv(x)
        x = x.transpose(1, -1)  # Channel swap
        x = self.ln(x)
        x = self.mlp(x)
        x = self.gamma * x
        x = x.transpose(1, -1)  # Channel swap
        return x + input


class ConvNeXt(nn.Module):
    """
    ConvNeXt  `A ConvNet for the 2020s` (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, input_shape, dims=None, depths=None, output_shape=None):
        super().__init__()

        self.input_shape = input_shape
        output_dim = Utils.prod(output_shape)
        channels_in = input_shape[0]

        if dims is None:
            dims = [96, 192, 32]

        dims = [channels_in] + dims

        if depths is None:
            depths = [1, 1, 3]

        self.ConvNeXt = nn.Sequential(*[nn.Sequential(nn.Conv2d(dims[i],
                                                                dims[i + 1],
                                                                kernel_size=4 if i == 0 else 2,
                                                                stride=4 if i == 0 else 2),  # Conv
                                                      nn.Sequential(Utils.ChannelSwap(),
                                                                    nn.LayerNorm(dims[i + 1]),
                                                                    Utils.ChannelSwap()) if i < 3
                                                      else nn.Identity(),  # LayerNorm
                                                      *[ConvNeXtBlock(dims[i + 1])
                                                        for _ in range(depth)])  # Conv, MLP, Residuals
                                        for i, depth in enumerate(depths)])

        self.project = nn.Identity() if output_dim is None \
            else nn.Sequential(AvgPool(), nn.Linear(dims[-1], output_dim))

        def weight_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

        self.apply(weight_init)

    def repr_shape(self, *_):
        return Utils.cnn_feature_shape(_, self.ConvNeXt, self.project)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = broadcast(self.input_shape, x)

        x = self.ConvNeXt(x)
        x = self.project(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out


class ConvNeXtTiny(ConvNeXt):
    def __init__(self, input_shape, output_dim=None):
        super().__init__(input_shape, [96, 192, 384, 768], [3, 3, 9, 3], output_dim)
