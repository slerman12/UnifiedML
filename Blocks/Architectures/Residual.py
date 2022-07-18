# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from torch import nn

from omegaconf import OmegaConf

import Utils


class Residual(nn.Module):
    """
    Residual with support for command-line instantiation
    """
    def __init__(self, model, down_sample=None, **kwargs):
        super().__init__()

        self.model = Utils.instantiate(OmegaConf.create({'_target_': model, **kwargs})) if isinstance(model, str) \
            else model

        self.down_sample = down_sample  # No command-line instantiation for down-sample

    def repr_shape(self, channels, height, width):
        return Utils.cnn_feature_shape([channels, height, width], self.model)

    def forward(self, input):
        output = self.model(input)

        if self.down_sample is not None:
            input = self.down_sample(input)

        return output + input  # Residual
