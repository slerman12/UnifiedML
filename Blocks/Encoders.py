# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import copy

import torch
from torch import nn

from Blocks.Architectures.Vision.CNN import CNN

import Utils


class CNNEncoder(nn.Module):
    """
    CNN encoder, e.g., DrQV2 (https://arxiv.org/abs/2107.09645).
    Generalized to multi-dimensionality convolutions and obs shapes (1d or 2d)
    """

    def __init__(self, obs_spec, context_dim=0, standardize=False, norm=False, eyes=None, pool=None, parallel=False,
                 optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.obs_spec = obs_spec

        self.standardize = \
            standardize and None not in [obs_spec.mean, obs_spec.stddev]  # Whether to center-scale (0 mean, 1 stddev)
        self.normalize = norm and None not in [obs_spec.low, obs_spec.high]  # Whether to [0, 1] shift-max scale

        self.mean, self.stddev = map(lambda stat: None if stat is None else torch.as_tensor(stat),
                                     (obs_spec.mean, obs_spec.stddev))

        # Dimensions
        obs_spec.shape[0] += context_dim

        # CNN
        self.Eyes = nn.Sequential(Utils.instantiate(eyes, input_shape=obs_spec.shape) or CNN(obs_spec.shape))

        adapt_cnn(self.Eyes, obs_spec.shape)  # Adapt 2d CNN kernel sizes for 1d or small-d compatibility

        if parallel:
            self.Eyes = nn.DataParallel(self.Eyes)  # Parallel on visible GPUs

        self.feature_shape = Utils.cnn_feature_shape(*obs_spec.shape, self.Eyes)  # Feature map shape

        self.pool = Utils.instantiate(pool, input_shape=self.feature_shape) or nn.Flatten()

        self.repr_shape = Utils.cnn_feature_shape(*self.feature_shape, self.pool)  # Shape after pooling

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).eval()

    def update_ema_params(self):
        Utils.param_copy(self, self.ema, self.ema_decay)

    def forward(self, obs, *context, pool=True):
        # Operate on last 3 dims, then restore

        obs_shape = obs.shape  # Preserve leading dims
        obs = obs.flatten(0, -4)  # Flatten up to last 3 dims

        assert tuple(obs_shape[-3:]) == self.obs_spec.shape, f'Encoder received an invalid obs shape ' \
                                                             f'{tuple(obs_shape[1:])}, ≠ {self.obs_spec.shape}'

        axes = (1,) * len(obs.shape[2:])  # Spatial axes, useful for dynamic input shapes

        # Standardize/normalize pixels
        if self.standardize:
            obs = (obs - self.mean.to(obs.device).view(1, -1, *axes)) / self.stddev.to(obs.device).view(1, -1, *axes)
        elif self.normalize:
            obs = 2 * (obs - self.obs_spec.low) / (self.obs_spec.high - self.obs_spec.low) - 1

        # Optionally append a 1D context to channels, broadcasting
        obs = torch.cat([obs, *[c.reshape(obs.shape[0], c.shape[-1], *axes).expand(-1, -1, *obs.shape[2:])
                                for c in context]], 1)

        # CNN encode
        h = self.Eyes(obs)

        feature_shape = tuple(h.shape[-4:][1:] + (1,) * (3 - len(h.shape[-4:][1:])))  # Add spatial dims

        assert feature_shape == self.feature_shape, f'pre-computed feature_shape does not match feature shape ' \
                                                    f'{self.feature_shape}≠{feature_shape}'

        if pool:
            h = self.pool(h)
            assert h.shape[-1] == math.prod(self.repr_shape) or tuple(h.shape[-3:]) == self.repr_shape, \
                f'pre-computed repr_dim/repr_shape does not match output dim ' \
                f'{math.prod(self.repr_shape)}≠{h.shape[-1]}, {self.repr_shape}≠{tuple(h.shape[-3:])}'

        # Restore leading dims
        h = h.view(*obs_shape[:-3], *h.shape[1:])

        return h


# Adapts a 2d CNN to a smaller dimensionality (in case an image's spatial dim < kernel size)
def adapt_cnn(block, obs_shape):
    if isinstance(block, (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
        # Represent hyper-params as tuples
        if not isinstance(block.kernel_size, tuple):
            block.kernel_size = (block.kernel_size, block.kernel_size)
        if not isinstance(block.padding, tuple):
            block.padding = (block.padding, block.padding)

        # Set them to adapt to obs shape (2D --> 1D, etc) via contracted kernels / suitable padding
        block.kernel_size = tuple(min(kernel, obs) for kernel, obs in zip(block.kernel_size, obs_shape[-2:]))
        block.padding = tuple(0 if obs <= pad else pad for pad, obs in zip(block.padding, obs_shape[-2:]))

        # Contract the CNN kernels accordingly
        if isinstance(block, nn.Conv2d):
            block.weight = nn.Parameter(block.weight[:, :, :block.kernel_size[0], :block.kernel_size[1]])
    elif hasattr(block, 'modules'):
        for layer in block.children():
            # Iterate through all layers
            adapt_cnn(layer, obs_shape[-3:])
            obs_shape = Utils.cnn_feature_shape(*obs_shape[-3:], layer)
