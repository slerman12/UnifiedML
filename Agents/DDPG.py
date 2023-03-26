# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math

import torch

from Blocks.Creator import MonteCarlo

from Agents import DrQV2Agent


class DDPGAgent(DrQV2Agent):
    """
    Deep Deterministic Policy Gradient
    (https://arxiv.org/pdf/1509.02971.pdf)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.actor.creator.policy = DDPGPolicy


dev = None


class DDPGPolicy(MonteCarlo):
    """
    Generalized DDPG policy compatible with discrete spaces and even learnable stddev/entropy.

    AC2 example:
    python Run.py Policy=Agents.DDPG.DDPGPolicy
    """
    def __init__(self, action_spec, discrete, mean, *args, max_sigma=0.3, min_sigma=0.3, theta=0.15, **kwargs):
        super().__init__(action_spec, discrete, mean, *args, **kwargs)

        """
        Ornstein-Uhlenbeck Process
        """
        sigma = max_sigma - (max_sigma - min_sigma) * (1 - self.stddev)  # Constant at max_sigma by default

        action_dim = math.prod(action_spec.shape)

        global dev

        if dev is None:
            dev = 1 if self.discrete else torch.ones([action_dim], device=mean.device)

        # Technically, only sigma should be rand of distribution, e.g. should be rand_clip = 0 in Distribution  TODO
        # And scale must be positive should be... I don't know, for entropy, prob etc. maybe self.dev = and inexact prob
        dev -= theta * dev + sigma * torch.randn(action_dim, device=mean.device)

        self.stddev = dev
