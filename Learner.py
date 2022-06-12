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
    - this module defined within the pytorch module being optimized such that model.train() and model.eval() inherit
    """
    def __init__(self, optim):
        super().__init__()

        self.optim = optim
        self.lr = self.optim.defaults['lr']

        for param in self.optim.param_groups:
            param['lr'] = 1

        self.decoys = self.params = self._decoys = self.dists = self.running_sum = self.elite_score = None

        # Torch initializes grads to None by default
        for param_group in self.optim.param_groups:
            for params in param_group.values():
                if isinstance(params, list):
                    for param in params:
                        param.grad = torch.zeros_like(param.data)

        self.step()

    def propagate(self, loss, previous_loss, batch_size, retain_graph=False):
        uninitialized = previous_loss.isnan()
        # previous_loss[uninitialized] = 2 * loss[uninitialized]
        previous_loss[uninitialized] = loss[uninitialized]

        advantage = torch.relu(previous_loss - loss).mean() * batch_size
        self.running_sum += advantage

        # update = False
        # if advantage > self.elite_score and not retain_graph:
        #     self.elite_score = advantage
        #     update = True

        for decoy, param, _decoy, dist in zip(self.decoys, self.params, self._decoys, self.dists):
            decoy.grad += (param.data - decoy.data) * advantage
            # if update:
            #     decoy.grad = param.data - decoy.data
            if not retain_graph:
                decoy.data = dist.sample()
                decoy.data = torch.sign(decoy.data) * self.lr  # Added this (can remove lr altogether this way)
                _decoy.data = decoy.data

    def step(self):
        if self.running_sum:

            for decoy in self.decoys:
                # decoy.grad /= self.running_sum
                decoy.grad /= max(self.running_sum, 1)
            self.optim.step()

        else:
            self.decoys = sum([param_group[key]
                               for param_group in self.optim.param_groups
                               for key in param_group if isinstance(param_group[key], list)], [])

        self._decoys = deepcopy(self.decoys)

        self.params = deepcopy(self.decoys)

        self.dists = []

        for param in self.params:
            dist = torch.distributions.Normal(param.data, self.lr)
            self.dists.append(dist)

        self.running_sum = self.elite_score = 0

    def zero_grad(self, **kwargs):
        self.optim.zero_grad()

    def train(self, mode=True):
        super().train(mode)
        for decoy, param, _decoy in zip(self.decoys, self.params, self._decoys):
            decoy.data = _decoy.data if mode else param.data

