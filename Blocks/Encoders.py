# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import copy
import math

import torch
from torch import nn

from Blocks.Architectures import MLP
from Blocks.Architectures.Vision.CNN import CNN

import Utils


class CNNEncoder(nn.Module):
    """
    CNN encoder, e.g., DrQV2 (https://arxiv.org/abs/2107.09645).
    Generalized to multi-dimensionality convolutions and obs shapes (1d or 2d)
    Isotropic here means dimensionality conserving
    """

    def __init__(self, obs_shape, context_dim=0, data_norm=None, shift_max_norm=False, isotropic=False, parallel=False,
                 eyes=None, pool=None, optim=None, scheduler=None, lr=0, lr_decay_epochs=0,
                 weight_decay=0, ema_decay=0):

        super().__init__()

        self.obs_shape = copy.copy(obs_shape)
        self.data_norm = torch.tensor(data_norm or [127.5, 255]).view(2, 1, -1, 1, 1)

        # Dimensions
        self.in_channels = obs_shape[0] + context_dim
        self.out_channels = obs_shape[0] if isotropic else 32  # Default 32

        # CNN
        self.Eyes = nn.Sequential(Utils.instantiate(eyes, input_shape=self.obs_shape)
                                  or CNN(self.in_channels, self.out_channels, depth=3),
                                  Utils.ShiftMaxNorm(-3) if shift_max_norm else nn.Identity())

        adapt_cnn(self.Eyes, self.obs_shape)  # Adapt 2d CNN kernel sizes for 1d or small-d compatibility

        if parallel:
            self.Eyes = nn.DataParallel(self.Eyes)  # Parallel on visible GPUs

        self.feature_shape = Utils.cnn_feature_shape(*self.obs_shape, self.Eyes)  # Feature map shape

        self.pool = Utils.instantiate(pool, input_shape=self.feature_shape) or nn.Flatten()

        self.repr_shape = Utils.cnn_feature_shape(*self.feature_shape, self.pool)
        self.repr_dim = math.prod(self.repr_shape)  # Flattened repr dim

        # Initialize model
        self.init(optim, scheduler, lr, lr_decay_epochs, weight_decay, ema_decay)

        # Isotropic
        if isotropic:
            assert tuple(obs_shape) == self.feature_shape, \
                f'specified to be isotropic, but in ≠ out {tuple(obs_shape)} ≠ {self.feature_shape}'

    def init(self, optim=None, scheduler=None, lr=None, lr_decay_epochs=0, weight_decay=0, ema_decay=0):
        # Optimizer
        if lr or Utils.can_instantiate(optim):
            self.optim = Utils.instantiate(optim, params=self.parameters(), lr=getattr(optim, 'lr', lr)) \
                         or torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Learning rate scheduler
        if lr_decay_epochs or Utils.can_instantiate(scheduler):
            self.scheduler = Utils.instantiate(scheduler, optimizer=self.optim) \
                             or torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, lr_decay_epochs)

        # EMA
        if ema_decay:
            self.ema, self.ema_decay = copy.deepcopy(self).eval(), ema_decay

    def update_ema_params(self):
        assert hasattr(self, 'ema')
        Utils.param_copy(self, self.ema, self.ema_decay)

    # Encodes
    def forward(self, obs, *context, pool=True):
        obs_shape = obs.shape  # Preserve leading dims
        obs = obs.flatten(0, -4)  # Encode last 3 dims

        assert obs_shape[-3:] == self.obs_shape, f'encoder received an invalid obs shape {obs_shape}'

        # Normalizes pixels
        mean, stddev = self.data_norm = self.data_norm.to(obs.device)
        obs = (obs - mean) / stddev

        # Optionally append context to channels assuming dimensions allow
        context = [c.reshape(obs.shape[0], c.shape[-1], 1, 1).expand(-1, -1, *self.obs_shape[1:])
                   for c in context]
        obs = torch.cat([obs, *context], 1)

        # CNN encode
        h = self.Eyes(obs)

        assert tuple(h.shape[-3:]) == self.feature_shape, f'pre-computed feature_shape does not match feature shape ' \
                                                          f'{self.feature_shape}≠{tuple(h.shape[-3:])}'

        if pool:
            h = self.pool(h)
            assert h.shape[-1] == self.repr_dim or tuple(h.shape[-3:]) == self.repr_shape, \
                f'pre-computed repr_dim/repr_shape does not match output dim ' \
                f'{self.repr_dim}≠{h.shape[-1]}, {self.repr_shape}≠{tuple(h.shape[-3:])}'

        # Restore leading dims
        h = h.view(*obs_shape[:-3], *h.shape[1:])
        return h


# Adapts a 2d CNN to a smaller dimensionality (in case an image's spatial dim < kernel size)
def adapt_cnn(block, obs_shape):
    if isinstance(block, (nn.Conv2d, nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
        # Represent conv hyper-params as tuples
        if not isinstance(block.kernel_size, tuple):
            block.kernel_size = (block.kernel_size, block.kernel_size)

        # Set them to adapt to obs shape (2D --> 1D, etc)
        block.kernel_size = tuple(min(kernel, obs) for kernel, obs in zip(block.kernel_size, obs_shape[-2:]))

        # Update the CNN architecture accordingly
        if not isinstance(block, nn.MaxPool2d):
            block.weight = nn.Parameter(block.weight[:, :, :block.kernel_size[0], :block.kernel_size[1]])
    elif hasattr(block, 'modules'):
        for layer in block.children():
            adapt_cnn(layer, obs_shape[-3:])
            obs_shape = Utils.cnn_feature_shape(*obs_shape[-3:], layer)


class MLPEncoder(nn.Module):
    """MLP encoder:
    With LayerNorm
    Can also l2-normalize penultimate layer (https://openreview.net/pdf?id=9xhgmsNVHu)"""

    def __init__(self, input_dim, output_dim, hidden_dim=512, depth=1, data_norm=None, layer_norm_dim=None,
                 non_linearity=nn.ReLU(inplace=True), dropout=0, binary=False, l2_norm=False,
                 parallel=False, optim=None, scheduler=None, lr=0, lr_decay_epochs=0, weight_decay=0, ema_decay=0):
        super().__init__()

        self.data_norm = torch.tensor(data_norm or [0, 1])

        self.trunk = nn.Identity() if layer_norm_dim is None \
            else nn.Sequential(nn.Linear(input_dim, layer_norm_dim),
                               nn.LayerNorm(layer_norm_dim),
                               nn.Tanh())

        # Dimensions
        in_features = layer_norm_dim or input_dim

        # MLP
        self.MLP = MLP(in_features, output_dim, hidden_dim, depth, non_linearity, dropout, binary, l2_norm)

        if parallel:
            self.MLP = nn.DataParallel(self.MLP)  # Parallel on visible GPUs

        self.init(optim, scheduler, lr, lr_decay_epochs, weight_decay, ema_decay)

        self.repr_shape, self.repr_dim = (output_dim,), output_dim

    def init(self, optim=None, scheduler=None, lr=None, lr_decay_epochs=0, weight_decay=0, ema_decay=None):
        # Optimizer
        if lr or Utils.can_instantiate(optim):
            self.optim = Utils.instantiate(optim, params=self.parameters(), lr=getattr(optim, 'lr', lr)) \
                         or torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Learning rate scheduler
        if lr_decay_epochs or Utils.can_instantiate(scheduler):
            self.scheduler = Utils.instantiate(scheduler, optimizer=self.optim) \
                             or torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, lr_decay_epochs)

        # EMA
        if ema_decay:
            self.ema, self.ema_decay = copy.deepcopy(self).eval(), ema_decay

    def update_ema_params(self):
        assert hasattr(self, 'ema_decay')
        Utils.param_copy(self, self.ema, self.ema_decay)

    def forward(self, x, *context):
        h = torch.cat([x, *context], -1)

        # Normalizes
        mean, stddev = self.data_norm = self.data_norm.to(h.device)
        h = (h - mean) / stddev

        h = self.trunk(h)
        return self.MLP(h)
