# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import copy

import torch
from torch import nn

import Utils


class Residual(nn.Module):
    def __init__(self, module, down_sample=None):
        super().__init__()
        self.module = module
        self.down_sample = down_sample

    def forward(self, x):
        y = self.module(x)
        if self.down_sample is not None:
            x = self.down_sample(x)
        return y + x

    def output_shape(self, height, width):
        return Utils.cnn_output_shape(height, width, self.module)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None, optim_lr=None, target_tau=None):
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

        self.init(optim_lr, target_tau)

    def init(self, optim_lr=None, target_tau=None):
        # Initialize weights
        self.apply(Utils.weight_init)

        # Optimizer
        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        # EMA
        if target_tau is not None:
            self.target = copy.deepcopy(self)
            self.target_tau = target_tau

    def update_target_params(self):
        assert hasattr(self, 'target_tau')
        Utils.param_copy(self, self.target, self.target_tau)

    def forward(self, x):
        return self.Residual_block(x)

    def output_shape(self, height, width):
        return Utils.cnn_output_shape(height, width, self.Residual_block)


