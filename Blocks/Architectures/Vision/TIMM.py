# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from torch import nn
import timm

from Blocks.Architectures.Vision.CNN import cnn_broadcast

import Utils


class TIMM(nn.Module):
    """
    Backwards compatibility with the TIMM (Pytorch Image Models) ecosystem. Download or load a model as follows.
    Usage: python Run.py  task=classify/mnist  Eyes='TIMM("ResNet18",pretrained=True)'
    """
    def __init__(self, name, pretrained=False, input_shape=None, output_shape=None, pool='avg'):
        super().__init__()

        self.input_shape, output_dim = Utils.to_tuple(input_shape), Utils.prod(output_shape)

        in_channels = self.input_shape[0]

        self.model = timm.create_model(name, pretrained=pretrained, in_chans=in_channels,
                                       num_classes=0 if output_shape is None else output_dim,
                                       global_pool='' if output_shape is None else pool)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = cnn_broadcast(self.input_shape, x)

        x = self.model(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out
