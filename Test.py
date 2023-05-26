# from torch.utils.data import DataLoader
# from torchvision.datasets.mnist import MNIST
# from torchvision.transforms import ToTensor
#
#
# dataset = MNIST('./', download=True, transform=ToTensor())
#
# dataset = DataLoader(dataset=dataset,
#                      pin_memory=True)  # pin_memory triggers CUDA error
#
# for _ in dataset:
#     continue

# import torch
# from torch import nn
# from torch.optim import SGD, Adam
#
# torch.manual_seed(0)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(0)
#
# bce = nn.BCELoss()
#
# a = torch.rand([10])
# b = torch.rand([10])
#
# model1 = nn.Linear(10, 10)
# model2 = nn.Linear(10, 10)
#
# # optim = SGD(list(model1.parameters()) + list(model2.parameters()), lr=1e-4)
# optim = Adam(list(model1.parameters()) + list(model2.parameters()), lr=0.0002, betas=(0.5, 0.999))
#
# y1 = nn.Sigmoid()(model1(a))
# y2 = nn.Sigmoid()(model2(b))
#
# ones = torch.ones([20])
# bce(torch.cat([y1, y2], 0), ones).backward()
# grad1 = model1.weight.grad
# grad2 = model2.weight.grad
#
# optim.zero_grad()
# y1 = nn.Sigmoid()(model1(a))
# y2 = nn.Sigmoid()(model2(b))
# ones = torch.ones([10])
# ((bce(y1, ones) + bce(y2, ones)) / 2).backward()
#
# assert torch.allclose(model1.weight.grad, grad1)
# assert torch.allclose(model2.weight.grad, grad2)


# Full reproducible playground to test this yourself


# def main():
#     yield 5
#     print('yes')


# i = iter(main())
# print(next(i))
#
# try:
#     next(i)
# except StopIteration:
#     pass

# print(list([*main()]))

import multiprocessing
import random
import time
import torch
import torch.multiprocessing as mp
import tensordict
from tensordict import TensorDict
import numpy as np


def f1(a):
    while True:
        _start = time.time()
        print(a[0]['hi'][0, 0, 0].item(), time.time() - _start, 'get')
        time.sleep(3)


def f2(a):
    while True:
        _start = time.time()
        print(a[0]['hi'][0, 0, 0].item(), time.time() - _start, 'get')
        time.sleep(3)


class Memory:
    def __init__(self):
        manager = mp.Manager()
        self.exps = [manager.list()]
        self.episode = manager.list()
        # self.exps = []
        # self.episode = []

        self.done = True

    def add(self, exp):
        # Truly shared memory
        for key in exp:
            exp[key] = torch.as_tensor(exp[key]).share_memory_()

        batch_size = max(len(mem) for mem in exp.values() if getattr(mem, 'shape', None))

        if exp['done']:
            # index = torch.as_tensor(tuple(enumerate([len(self.exps)] * batch_size)), dtype=torch.int).share_memory_()
            index = enumerate([len(self.exps)] * batch_size)  # Faster I think
            self.episode.extend(index)

        # self.lengths

        self.done = exp['done']

        self.exps.append(exp)

    def initialize_worker(self):
        # if self.offline:  # Can cache locally
        #     self.exps, self.episode = map(list, (self.exps, self.episode))
        #     or self.cache = by index and corresponding episode index that indexes into it  TODO
        pass

    def sample(self):
        return self[random.randint(0, len(self))]

    def __getitem__(self, ind):
        mem_ind, exp_ind = self.episode[ind]

        # Sample across episode lengths

        return {key: value[mem_ind] if getattr(value, 'shape', None) else value
                for key, value in self.exps[exp_ind].items()}

    def __setitem__(self, ind, value):
        mem_ind, exp_ind = self.episode[ind]

        for mem in self.exps[exp_ind].values():
            if getattr(mem, 'shape', None):
                mem[mem_ind] = value

    def __len__(self):
        return len(self.episode)


if __name__ == '__main__':
    m = Memory()
    d = {'hi': np.ones([20000, 3, 32, 32]), 'done': True}
    start = time.time()
    m.add(d)
    print(time.time() - start, 'add')
    p1 = mp.Process(name='p1', target=f1, args=(m,))
    p2 = mp.Process(name='p', target=f2, args=(m,))
    p1.start()
    p2.start()
    start = time.time()
    m[0] = 0
    print(time.time() - start, 'set')
    p1.join()
    p2.join()


from functools import reduce
from operator import getitem


users = {
    'freddy': {
        'name': {
            'first': 'fred',
            'last': 'smith'
        },
        'postIds': [1, 2, 3]
    }
}
reduce(getitem, ['freddy', 'name', 'last'], initial=users)
reduce(getitem, ['freddy', 'postIds', 1], initial=users)
