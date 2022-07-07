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
    Isotropic here means dimensionality conserving
    """

    def __init__(self, obs_spec, context_dim=0, standardize=False, norm=False,
                 device='cuda', parallel=False, eyes=None, pool=None, isotropic=False,
                 optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):

        super().__init__()

        self.obs_shape = torch.Size(obs_spec.shape)

        self.standardize = \
            standardize and None not in [obs_spec.mean, obs_spec.stddev]  # Whether to center-scale (0 mean, 1 stddev)
        self.normalize = norm and None not in [obs_spec.low, obs_spec.high]  # Whether to [0, 1] shift-max scale

        if self.standardize:
            self.mean, self.stddev = Utils.to_torch((obs_spec.mean, obs_spec.stddev), device)

        self.low, self.high = obs_spec.low, obs_spec.high

        # Dimensions
        obs_spec.shape[0] += context_dim  # TODO no context dim for isotropic?
        self.out_channels = obs_spec.shape[0] if isotropic else 32  # Default 32

        # CNN
        self.Eyes = nn.Sequential(Utils.instantiate(eyes, input_shape=obs_spec.shape)
                                  or CNN(obs_spec.shape, self.out_channels, depth=3))

        adapt_cnn(self.Eyes, obs_spec.shape)  # Adapt 2d CNN kernel sizes for 1d or small-d compatibility

        if parallel:
            self.Eyes = nn.DataParallel(self.Eyes)  # Parallel on visible GPUs

        self.feature_shape = Utils.cnn_feature_shape(*obs_spec.shape, self.Eyes)  # Feature map shape

        self.pool = Utils.instantiate(pool, input_shape=self.feature_shape) or nn.Flatten()

        self.repr_shape = Utils.cnn_feature_shape(*self.feature_shape, self.pool)

        # Isotropic
        if isotropic:
            assert tuple(obs_spec.shape) == self.feature_shape, \
                f'specified to be isotropic, but in ≠ out {tuple(obs_spec.shape)} ≠ {self.feature_shape}'

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).eval()

    def update_ema_params(self):
        Utils.param_copy(self, self.ema, self.ema_decay)

    # Encodes
    def forward(self, obs, *context, pool=True):
        obs_shape = obs.shape  # Preserve leading dims
        obs = obs.flatten(0, -4)  # Encode last 3 dims

        assert obs_shape[-3:] == self.obs_shape, f'encoder received an invalid obs shape ' \
                                                 f'{obs_shape[1:]}, ≠ {self.obs_shape}'

        # Standardizes/normalizes pixels
        if self.standardize or self.normalize:
            obs = (obs - self.mean.view(1, -1, 1, 1)) / self.stddev.view(1, -1, 1, 1) if self.standardize \
                else 2 * (obs - self.low) / (self.high - self.low) - 1

        # Optionally append context to channels assuming dimensions allow
        context = [c.reshape(obs.shape[0], c.shape[-1], 1, 1).expand(-1, -1, *self.obs_shape[1:])
                   for c in context]
        obs = torch.cat([obs, *context], 1)

        # CNN encode
        h = self.Eyes(obs)

        feature_shape = list(self.feature_shape)
        for _ in range(len(self.feature_shape) - len(h.shape[-4:][1:])):
            feature_shape.remove(1)  # Feature shape w/o 1s

        assert list(h.shape[-4:][1:]) == feature_shape, f'pre-computed feature_shape does not match feature shape ' \
                                                        f'{feature_shape}≠{list(h.shape[-4:][1:])}'

        # assert tuple(h.shape[-3:]) == self.feature_shape, f'pre-computed feature_shape does not match feature shape ' \
        #                                                   f'{self.feature_shape}≠{tuple(h.shape[-3:])}'

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