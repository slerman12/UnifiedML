# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch.distributions import Normal, Categorical
from torch.distributions.utils import _standard_normal

import Utils


class TruncatedNormal(Normal):
    """
    A Gaussian Normal distribution generalized to multi-action sampling and the option to clip standard deviation.
    Consistent with torch.distributions.Normal
    """
    def __init__(self, loc, scale, low=None, high=None, eps=1e-6, stddev_clip=None):
        super().__init__(loc, scale)

        self.low, self.high = low, high
        self.eps = eps
        self.stddev_clip = stddev_clip

    def log_prob(self, value):
        if value.shape[-len(self.loc.shape):] == self.loc.shape:
            return super().log_prob(value)
        else:
            diff = len(value.shape) - len(self.loc.shape)
            return super().log_prob(value.transpose(0, diff)).transpose(0, diff)  # To account for batch_first=True

    # No grad, defaults to no clip, batch dim first
    def sample(self, sample_shape=1, to_clip=False, batch_first=True):
        with torch.no_grad():
            return self.rsample(sample_shape, to_clip, batch_first)

    def rsample(self, sample_shape=1, to_clip=True, batch_first=True):
        if isinstance(sample_shape, int):
            sample_shape = torch.Size((sample_shape,))

        # Draw multiple samples
        shape = self._extended_shape(sample_shape)

        rand = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)  # Explore
        dev = rand * self.scale.expand(shape)  # Deviate

        if to_clip:
            dev = Utils.rclamp(dev, -self.stddev_clip, self.stddev_clip)  # Don't explore /too/ much
        x = self.loc.expand(shape) + dev

        if batch_first:
            x = x.transpose(0, len(sample_shape))  # Batch dim first

        if self.low is not None and self.high is not None:
            # Differentiable truncation
            return Utils.rclamp(x, self.low + self.eps, self.high - self.eps)

        return x


class NormalizedCategorical(Categorical):
    """
    A Categorical that normalizes samples, allows sampling along specific "dim"s, and can temperature-weigh the softmax.
    Consistent with torch.distributions.Categorical
    """
    def __init__(self, probs=None, logits=None, low=None, high=None, temp=1, dim=-1):
        if probs is not None:
            probs = probs.transpose(-1, dim)

        if logits is not None:
            logits = logits.transpose(-1, dim)

        super().__init__(probs, logits / temp)

        self.low, self.high = low, high  # Range to normalize to
        self.dim = dim

        self.best = self.normalize(logits.argmax(-1, keepdim=True).transpose(-1, self.dim))

    def rsample(self, sample_shape=1, batch_first=True):
        self.sample(sample_shape, batch_first)  # Note: not differentiable

    def sample(self, sample_shape=1, batch_first=True):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        sample = super().sample(sample_shape)

        if batch_first:
            sample = sample.transpose(0, len(sample_shape))  # Batch dim first

        sample = sample.transpose(-1, self.dim)

        return self.normalize(sample)

    def normalize(self, sample):
        # Normalize -> [low, high]
        return sample / (self.logits.shape[-1] - 1) * (self.high - self.low) + self.low if self.low or self.high \
            else sample
