# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from torch import nn


class F_ckGradientDescent(nn.Module):
    """
    Sampling-based Hebbian method, a biologically-plausible optimizer
    * With all due respect to gradient descent, my favorite algorithm of all time.
    """
    def __init__(self, optim, step_optim_per_steps=10):
        super().__init__()

        self.optim = optim

        self.step = 0
        self.step_optim_per_steps = step_optim_per_steps

    def propagate(self, retain_graph=False):
        pass

    def step(self):
        pass
