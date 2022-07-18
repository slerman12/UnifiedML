# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Blocks.Architectures.MLP import MLP

from Blocks.Architectures.Vision.CNN import CNN

from Blocks.Architectures.Residual import Residual

from Blocks.Architectures.Vision.ResNet import MiniResNet, MiniResNet as ResNet, ResNet18, ResNet50
from Blocks.Architectures.Vision.ConvMixer import ConvMixer
from Blocks.Architectures.Vision.ConvNeXt import ConvNeXt, ConvNeXtTiny

from Blocks.Architectures.MultiHeadAttention import Attention, MHDPA, CrossAttention, SelfAttention, ReLA

from Blocks.Architectures.Transformer import AttentionBlock, CrossAttentionBlock, SelfAttentionBlock, Transformer

from Blocks.Architectures.Vision.ViT import ViT

from Blocks.Architectures.Perceiver import Perceiver

from Blocks.Architectures.RN import RN

from Blocks.Architectures.Vision.CNN import AvgPool
from Blocks.Architectures.Vision.ViT import CLSPool
