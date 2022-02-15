# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Blocks.Architectures.MLP import MLP
from Blocks.Architectures.MultiHeadAttention import AttentionPool
from Blocks.Architectures.Vision.CNN import CNN
from Blocks.Architectures.Vision.ViT import ViT
from Blocks.Architectures.Vision.ResNet import MiniResNet
from Blocks.Architectures.Vision.ResNet import MiniResNet as ResNet
from Blocks.Architectures.Vision.ConvMixer import ConvMixer
from Blocks.Architectures.Vision.ConvNeXt import ConvNeXt


from torch import nn


class Null(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *x):
        return x
