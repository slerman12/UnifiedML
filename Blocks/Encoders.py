# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import copy

import torch
from torch import nn

from Blocks.Architectures.Vision.CNN import CNN

import Utils


class CNNEncoder(nn.Module):
    """
    CNN encoder generalized to work with proprioceptive recipes and multi-dimensionality convolutions (1d or 2d)
    """
    def __init__(self, obs_spec, context_dim=0, standardize=False, norm=False, Eyes=None, pool=None, parallel=False,
                 optim=None, scheduler=None, lr=None, lr_decay_epochs=None, weight_decay=None, ema_decay=None):
        super().__init__()

        self.obs_shape = getattr(obs_spec, 'shape', obs_spec)  # Allow spec or shape

        for key in ('mean', 'stddev', 'low', 'high'):
            setattr(self, key, None if getattr(obs_spec, key, None) is None else torch.as_tensor(obs_spec[key]))

        self.standardize = \
            standardize and None not in [self.mean, self.stddev]  # Whether to center-scale (0 mean, 1 stddev)
        self.normalize = norm and None not in [self.low, self.high]  # Whether to [0, 1] shift-max scale

        # Dimensions
        obs_shape = [1] * (2 - len(self.obs_shape)) + list(self.obs_shape)  # At least 1 channel dim and spatial dim
        obs_shape[0] += context_dim

        # CNN
        self.Eyes = Utils.instantiate(Eyes, input_shape=obs_shape) or CNN(obs_shape)

        adapt_cnn(self.Eyes, obs_shape)  # Adapt 2d CNN kernel sizes for 1d or small-d compatibility

        if parallel:
            self.Eyes = nn.DataParallel(self.Eyes)  # Parallel on visible GPUs

        self.feature_shape = Utils.cnn_feature_shape(obs_shape, self.Eyes)  # Feature map shape

        self.pool = Utils.instantiate(pool, input_shape=self.feature_shape) or nn.Flatten()

        self.repr_shape = Utils.cnn_feature_shape(self.feature_shape, self.pool)  # Shape after pooling

        # Initialize model optimizer + EMA
        self.optim, self.scheduler = Utils.optimizer_init(self.parameters(), optim, scheduler,
                                                          lr, lr_decay_epochs, weight_decay)
        if ema_decay:
            self.ema_decay = ema_decay
            self.ema = copy.deepcopy(self).eval()

    def forward(self, obs, *context, pool=True):
        # Operate on non-batch dims, then restore

        dims = len(self.obs_shape)

        batch_dims = obs.shape[:-dims]  # Preserve leading dims
        axes = (1,) * (dims - 1)  # Spatial axes, useful for dynamic input shapes

        # Standardize/normalize pixels
        if self.standardize:
            obs = (obs - self.mean.to(obs.device).view(-1, *axes)) / self.stddev.to(obs.device).view(-1, *axes)
        elif self.normalize:
            obs = 2 * (obs - self.low) / (self.high - self.low) - 1

        try:
            channel_dim = (1,) * (not axes)  # At least 1 channel dim and spatial dim
            obs = obs.reshape(-1, *channel_dim, *self.obs_shape)  # Validate shape, collapse batch dims
        except RuntimeError:
            raise RuntimeError('\nObs shape does not broadcast to pre-defined obs shape '
                               f'{tuple(obs.shape)}, ≠ {self.obs_shape}')

        # Optionally append a 1D context to channels, broadcasting
        obs = torch.cat([obs, *[c.reshape(obs.shape[0], c.shape[-1], *axes or (1,)).expand(-1, -1, *obs.shape[2:])
                                for c in context]], 1)

        # CNN encode
        h = self.Eyes(obs)

        try:
            h = h.view(h.shape[0], *self.feature_shape)  # Validate shape
        except RuntimeError:
            raise RuntimeError('\nFeature shape cannot broadcast to pre-computed feature_shape '
                               f'{tuple(h.shape[1:])}≠{self.feature_shape}')

        if pool:
            h = self.pool(h)
            try:
                h = h.view(h.shape[0], *self.repr_shape)  # Validate shape
            except RuntimeError:
                raise RuntimeError('\nOutput shape after pooling does not match pre-computed repr_shape '
                                   f'{tuple(h.shape[1:])}≠{self.repr_shape}')

        h = h.view(*batch_dims, *h.shape[1:])  # Restore leading dims
        return h


# Adapts a 2d CNN to a smaller dimensionality or truncates (in case an image's spatial dim < kernel size)
def adapt_cnn(block, obs_shape):
    obs_shape = tuple(obs_shape)

    # Nd
    name = type(block).__name__
    N = 2 if '2d' in name else 1 if '1d' in name else 0

    if N:
        for attr in ['kernel_size', 'padding', 'stride', 'dilation', 'output_padding', 'output_size']:
            if hasattr(block, attr):
                val = getattr(block, attr)

                # Represent hyper-params as tuples
                if not isinstance(val, tuple):
                    setattr(block, attr, (val,) * N)
                    val = getattr(block, attr)

                # Set to adapt to obs shape: contracted kernels, padding, etc
                setattr(block, attr, tuple(min(dim, adapt) for dim, adapt in zip(val, obs_shape[1:])))

        # Update operation to 1d if needed
        if 0 < len(obs_shape[1:]) < N:
            block.forward = getattr(nn, name.replace('2d', '1d')).forward.__get__(block)
        # Contract a 1d CNN's kernels
        if isinstance(block, nn.Conv1d):
            block.weight = nn.Parameter(block.weight[:, :, :block.kernel_size[0]])
        # Contract a 2d CNN's kernels
        elif isinstance(block, nn.Conv2d):
            truncate = [slice(0, block.kernel_size[i]) if i < len(obs_shape[1:]) else 0 for i in [0, 1]]  # 2D trunc
            block.weight = nn.Parameter(block.weight[:, :, truncate[0], truncate[1]])

            # Update operation to 1d if needed
            if len(block.kernel_size) == 1:
                block._conv_forward = nn.Conv1d._conv_forward.__get__(block, nn.Conv2d)
    elif hasattr(block, 'modules'):
        for layer in block.children():
            # Iterate through all layers
            adapt_cnn(layer, obs_shape)  # Dimensionality-adaptivity
            obs_shape = Utils.cnn_feature_shape(obs_shape, layer)
