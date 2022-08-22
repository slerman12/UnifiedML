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

    Algorithm: compare loss of randomly sampled weights to best previous loss for this exact batch data, then
    update the weights in proportion to the "advantage" (improvement since last) of the randomly sampled weights
    """
    def __init__(self, optim):
        super().__init__()

        self.optim = optim
        self.lr = self.optim.defaults['lr']

        for param in self.optim.param_groups:
            param['lr'] = 1

        self.decoys = self.params = self._decoys = self.samplers = None

        self.samplers = []

        # Torch initializes grads to None by default
        for param_group in self.optim.param_groups:
            for params in param_group.values():
                if isinstance(params, list):
                    for param in params:
                        # Initialize to 0, set sampler
                        param.grad = torch.zeros_like(param.data)
                        sampler = torch.distributions.Normal(torch.zeros_like(param), self.lr)
                        self.samplers.append(sampler)

        self.running_sum = self.elite_score = 0
        self.step()

    # called for every batch
    def propagate(self, loss, previous_loss, batch_size, retain_graph=False):
        # previous_loss is the previous errors w.r.t. each sample in the batch
        # NOT previous batch's loss
        previous_loss = deepcopy(previous_loss)

        uninitialized = previous_loss.isnan()
        previous_loss[uninitialized] = loss[uninitialized]

        # advantage = torch.relu(previous_loss - loss).sum()
        advantage = torch.relu(previous_loss.mean() - loss.mean())
        self.running_sum += advantage

        # update = False
        # if advantage > self.elite_score and not retain_graph:
        #     self.elite_score = advantage
        #     update = True

        for decoy, param, _decoy, sampler in zip(self.decoys, self.params, self._decoys, self.samplers):
            decoy.grad += (param.data - decoy.data) * advantage
            # if update:
            #     decoy.grad = param.data - decoy.data
            if not retain_graph:
                sample = sampler.sample()
                decoy.data = param.data + sample
                _decoy.data = decoy.data

    def step(self):
        # step() is called once every K batches
        if self.running_sum:

            for param, decoy in zip(self.params, self.decoys):
                decoy.data = param.data
                decoy.grad /= self.running_sum or 1
            self.optim.step()

        else:
            self.decoys = sum([param_group[key]
                               for param_group in self.optim.param_groups
                               for key in param_group if isinstance(param_group[key], list)], [])

        self._decoys = deepcopy(self.decoys)

        self.params = deepcopy(self.decoys)

    # called once every K batches
    def zero_grad(self, **kwargs):
        self.running_sum = self.elite_score = 0
        self.optim.zero_grad()

    def train(self, mode=True):
        super().train(mode)
        for decoy, param, _decoy in zip(self.decoys, self.params, self._decoys):
            decoy.data = _decoy.data if mode else param.data

