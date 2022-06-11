# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from copy import deepcopy

import torch


class F_ckGradientDescent(torch.nn.Module):
    """
    Sampling-based Hebbian method, a biologically-plausible NN optimizer
    * With all due respect to gradient descent, my favorite algorithm of all time.

    Accepts an existing torch optimizer as input but makes three assumptions:
    - loss.backward is never called
    - instead, this module's propagate method is called. It takes no derivatives, is fully parallelizable per weight
    - this module is defined within the pytorch module being optimized such that model.train() and model.eval() transfer
    """
    def __init__(self, optim):
        super().__init__()

        self.optim = optim

        self.decoys = self.params = self._decoys = self.dists = self.running_sum = None

        self.step()
        self.zero_grad()

    def propagate(self, loss, previous_loss, batch_size, retain_graph=False):
        uninitialized = previous_loss.isnan()
        previous_loss[uninitialized] = 2 * loss[uninitialized]

        advantage = torch.relu(loss - previous_loss) * batch_size
        self.running_sum += advantage

        for decoy, param, _decoy, dist in zip(self.decoys, self.params, self._decoys, self.dists):
            decoy.grad += (param.data - decoy.data) * advantage
            if not retain_graph:
                decoy.data = dist.sample()
                _decoy.data = decoy.data

    def step(self):
        if self.running_sum:

            for decoy in self.decoys:
                decoy.grad /= self.running_sum
            self.optim.step()

        self.decoys = sum([param_group[key]
                           for param_group in self.optim.param_groups
                           for key in param_group], [])

        self._decoys = deepcopy(self.decoys)

        self.params = deepcopy(self.decoys)

        self.dists = []

        for param in self.params:
            dist = torch.distributions.Normal(param.data, self.optim.defaults['lr'])
            self.dists.append(dist)

        self.running_sum = 0

    def zero_grad(self, **kwargs):
        self.optim.zero_grad()

    def train(self, mode=True):
        for decoy, param, _decoy in zip(self.decoys, self.params, self._decoys):
            decoy.data = _decoy.data if mode else param.data

