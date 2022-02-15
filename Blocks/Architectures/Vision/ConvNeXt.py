# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import copy

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

    def output_shape(self, h, w):
        return Utils.cnn_output_shape(h, w, self.conv)

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
    r""" ConvNeXt  `A ConvNet for the 2020s` https://arxiv.org/pdf/2201.03545.pdf
    Args:
        dims (list): Feature dimension at each stage, starting with in-channels. Default: [3, 96, 192, 384, 768]
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        num_classes (int): Number of classes for classification head. Default: 1000
    """
    def __init__(self, input_shape, dims=None, depths=None, num_classes=1000):
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
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Sequential(Utils.ChannelSwap(),
                                               nn.LayerNorm(dims[-1]),
                                               Utils.ChannelSwap()),  # LayerNorm
                                 # nn.Flatten(),
                                 # nn.Linear(dims[-1], num_classes)
                                 )

        self.init(None, None)

    def output_shape(self, h, w):
        return Utils.cnn_output_shape(h, w, self.CNN)

    def init(self, optim_lr, ema_tau):
        def weight_init(m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

        self.apply(weight_init)

        # Optimizer
        if optim_lr is not None:
            self.optim = torch.optim.AdamW(self.parameters(), lr=optim_lr,
                                           eps=1e-8, weight_decay=0.05)

        # EMA
        if ema_tau is not None:
            self.ema = copy.deepcopy(self)
            self.ema_tau = ema_tau

    def update_ema_params(self):
        assert hasattr(self, 'ema_tau')
        Utils.param_copy(self, self.ema, self.ema_tau)

    def forward(self, x):
        return self.CNN(x)
