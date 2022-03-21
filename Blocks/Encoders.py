# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import copy
import math

from hydra.utils import instantiate

import torch
from torch import nn

from Blocks.Architectures import MLP
from Blocks.Architectures.Vision.CNN import CNN
from Blocks.Architectures.Vision.ResNet import MiniResNet

import Utils


class CNNEncoder(nn.Module):
    """
    Basic CNN encoder, e.g., DrQV2 (https://arxiv.org/abs/2107.09645).
    """

    def __init__(self, obs_shape, out_channels=32, depth=3, data_norm=None, shift_max_norm=False,
                 recipe=None, parallel=False, lr=None, weight_decay=0, ema_decay=None):

        super().__init__()

        assert len(obs_shape) == 3, 'image observation shape must have 3 dimensions'

        self.obs_shape = obs_shape
        self.data_norm = torch.tensor(data_norm or [127.5, 255]).view(2, 1, -1, 1, 1)

        # CNN
        self.Eyes = nn.Sequential(CNN(obs_shape, out_channels, depth) if not (recipe and recipe.eyes._target_)
                                  else instantiate(recipe.eyes),
                                  Utils.ShiftMaxNorm(-3) if shift_max_norm else nn.Identity())
        if parallel:
            self.Eyes = nn.DataParallel(self.Eyes)  # Parallel on visible GPUs

        self.pool = nn.Flatten() if not (recipe and recipe.pool._target_) \
            else instantiate(recipe.pool, input_shape=self._feature_shape())

        # Initialize model
        self.init(lr, weight_decay, ema_decay)

    def _feature_shape(self):
        return Utils.cnn_feature_shape(*self.obs_shape, self.Eyes)

    def init(self, lr=None, weight_decay=0, ema_decay=None):
        # Optimizer
        if lr is not None:
            self.optim = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Dimensions
        self.feature_shape = self._feature_shape()  # Feature map shape

        self.repr_shape = Utils.cnn_feature_shape(*self.feature_shape, self.pool)
        self.repr_dim = math.prod(self.repr_shape)  # Flattened repr dim

        # EMA
        if ema_decay is not None:
            self.ema = copy.deepcopy(self).eval()
            self.ema_decay = ema_decay

    def update_ema_params(self):
        assert hasattr(self, 'ema')
        Utils.param_copy(self, self.ema, self.ema_decay)

    # Encodes
    def forward(self, obs, *context, pool=True):
        obs_shape = obs.shape  # Preserve leading dims
        assert obs_shape[-3:] == self.obs_shape, f'encoder received an invalid obs shape {obs_shape}'
        obs = obs.flatten(0, -4)  # Encode last 3 dims

        # Normalizes pixels
        mean, stddev = self.data_norm = self.data_norm.to(obs.device)
        obs = (obs - mean) / stddev

        # Optionally append context to channels assuming dimensions allow
        context = [c.reshape(obs.shape[0], c.shape[-1], 1, 1).expand(-1, -1, *self.obs_shape[1:])
                   for c in context]
        obs = torch.cat([obs, *context], 1)

        # CNN encode
        h = self.Eyes(obs)

        assert tuple(h.shape[-3:]) == self.feature_shape, f'pre-computed feature_shape does not match feature shape' \
                                                          f'{self.feature_shape}≠{tuple(h.shape[-3:])}'

        if pool:
            h = self.pool(h)
            assert h.shape[-1] == self.repr_dim or tuple(h.shape[-3:]) == self.repr_shape, \
                f'pre-computed repr_dim/repr_shape does not match output dim ' \
                f'{self.repr_dim}≠{h.shape[-1]}, {self.repr_shape}≠{tuple(h.shape[-3:])}'

        # Restore leading dims
        h = h.view(*obs_shape[:-3], *h.shape[1:])
        return h


class ResidualBlockEncoder(CNNEncoder):
    """
    Residual block-based CNN encoder,
    Isotropic means no bottleneck / dimensionality conserving
    """

    def __init__(self, obs_shape, context_dim=0, out_channels=32, hidden_channels=64, num_blocks=1, data_norm=None,
                 shift_max_norm=True, isotropic=False, recipe=None, parallel=False,
                 lr=None, weight_decay=0, ema_decay=None):

        super().__init__(obs_shape, hidden_channels, 0, data_norm)

        # Dimensions
        in_channels = obs_shape[0] + context_dim
        out_channels = obs_shape[0] if isotropic else out_channels

        # CNN ResNet-ish
        self.Eyes = nn.Sequential(MiniResNet((in_channels, *obs_shape[1:]), 3, 2 - isotropic,
                                             [hidden_channels, out_channels], [num_blocks]),
                                  Utils.ShiftMaxNorm(-3) if shift_max_norm else nn.Identity())
        if parallel:
            self.Eyes = nn.DataParallel(self.Eyes)  # Parallel on visible GPUs

        self.init(lr, weight_decay, ema_decay)

        # Isotropic
        if isotropic:
            assert tuple(obs_shape) == self.feature_shape, \
                f'specified to be isotropic, but in {tuple(obs_shape)} ≠ out {self.feature_shape}'


class MLPEncoder(nn.Module):
    """MLP encoder:
    With LayerNorm
    Can also l2-normalize penultimate layer (https://openreview.net/pdf?id=9xhgmsNVHu)"""

    def __init__(self, input_dim, output_dim, hidden_dim=512, depth=1, data_norm=None, layer_norm_dim=None,
                 non_linearity=nn.ReLU(inplace=True), dropout=0, binary=False, l2_norm=False,
                 parallel=False, lr=None, weight_decay=0, ema_decay=None):
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

        self.init(lr, weight_decay, ema_decay)

        self.repr_shape, self.repr_dim = (output_dim,), output_dim

    def init(self, lr=None, weight_decay=0, ema_decay=None):
        # Initialize weights
        self.apply(Utils.weight_init)

        # Optimizer
        if lr is not None:
            self.optim = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        # EMA
        if ema_decay is not None:
            self.ema = copy.deepcopy(self).eval()
            self.ema_decay = ema_decay

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
