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
import math
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

# import multiprocessing
# import random
# import time
# import torch
# import torch.multiprocessing as mp
# import tensordict
# from tensordict import TensorDict
# import numpy as np
#
#
# def f1(a):
#     while True:
#         _start = time.time()
#         print(a[0]['hi'][0, 0, 0].item(), time.time() - _start, 'get')
#         time.sleep(3)
#
#
# def f2(a):
#     while True:
#         _start = time.time()
#         print(a[0]['hi'][0, 0, 0].item(), time.time() - _start, 'get')
#         time.sleep(3)
#
#
# class Memory:
#     def __init__(self):
#         manager = mp.Manager()
#         self.exps = [manager.list()]
#         self.episode = manager.list()
#         # self.exps = []
#         # self.episode = []
#
#         self.done = True
#
#     def add(self, exp):
#         # Truly shared memory
#         for key in exp:
#             exp[key] = torch.as_tensor(exp[key]).share_memory_()
#
#         batch_size = max(len(mem) for mem in exp.values() if getattr(mem, 'shape', None))
#
#         if exp['done']:
#             # index = torch.as_tensor(tuple(enumerate([len(self.exps)] * batch_size)), dtype=torch.int).share_memory_()
#             index = enumerate([len(self.exps)] * batch_size)  # Faster I think
#             self.episode.extend(index)
#
#         # self.lengths
#
#         self.done = exp['done']
#
#         self.exps.append(exp)
#
#     def initialize_worker(self):
#         # if self.offline:  # Can cache locally
#         #     self.exps, self.episode = map(list, (self.exps, self.episode))
#         #     or self.cache = by index and corresponding episode index that indexes into it  TODO
#         pass
#
#     def sample(self):
#         return self[random.randint(0, len(self))]
#
#     def __getitem__(self, ind):
#         mem_ind, exp_ind = self.episode[ind]
#
#         # Sample across episode lengths
#
#         return {key: value[mem_ind] if getattr(value, 'shape', None) else value
#                 for key, value in self.exps[exp_ind].items()}
#
#     def __setitem__(self, ind, value):
#         mem_ind, exp_ind = self.episode[ind]
#
#         for mem in self.exps[exp_ind].values():
#             if getattr(mem, 'shape', None):
#                 mem[mem_ind] = value
#
#     def __len__(self):
#         return len(self.episode)
#
#
# if __name__ == '__main__':
#     m = Memory()
#     d = {'hi': np.ones([20000, 3, 32, 32]), 'done': True}
#     start = time.time()
#     m.add(d)
#     print(time.time() - start, 'add')
#     p1 = mp.Process(name='p1', target=f1, args=(m,))
#     p2 = mp.Process(name='p', target=f2, args=(m,))
#     p1.start()
#     p2.start()
#     start = time.time()
#     m[0] = 0
#     print(time.time() - start, 'set')
#     p1.join()
#     p2.join()
#
#
# from functools import reduce
# from operator import getitem
#
#
# users = {
#     'freddy': {
#         'name': {
#             'first': 'fred',
#             'last': 'smith'
#         },
#         'postIds': [1, 2, 3]
#     }
# }
# reduce(getitem, ['freddy', 'name', 'last'], initial=users)
# reduce(getitem, ['freddy', 'postIds', 1], initial=users)



import multiprocessing as mp
import os
from abc import abstractmethod

import torch

NCORE = 4

def process(q, iolock):
    from time import sleep
    while True:
        stuff = q.get()
        if stuff is None:
            break
        with iolock:
            print("processing", stuff)
        sleep(stuff)


class PersistentWorker:
    def __init__(self, worker, num_workers, map_args=True, **kwargs):
        self.main_worker = os.getpid()

        self.worker = worker
        self.num_workers = num_workers

        # For sending data to and from workers
        self.main_pipes, self.worker_pipes = zip(*[mp.Pipe() for _ in range(num_workers)])

        self.dones = [torch.tensor(False).share_memory_() for _ in range(num_workers)]
        self.dones[-1][...] = True

        self.map_args = map_args  # Can assume args or at least signal to go

        self.__dict__.update(kwargs)

    def __call__(self):
        while True:
            if (self.worker_pipes[self.worker].poll() or not self.map_args) \
                    and not self.main_pipes[self.worker].poll() \
                    and self.dones[self.worker - 1 if self.worker else -1]:
                args = self.worker_pipes[self.worker].recv() if self.map_args else ()
                outs = self.target(*args)
                self.worker_pipes[self.worker].send(outs)
                self.dones[self.worker - 1 if self.worker else -1][...] = False
                self.dones[self.worker][...] = True  # FULLY SEQUENTIAL!!! Iterate through all pipes and order by pipe

    @abstractmethod
    def target(self, *args):
        pass

    def map(self, args):
        assert self.main_worker == os.getpid(), 'Only main worker can call consume.'

        if self.map_args:
            for i in range(self.num_workers):
                self.main_pipes[i].send(args[i])

        outs = []
        while len(outs) < self.num_workers:
            if self.main_pipes[len(outs)].poll():
                outs.append(self.main_pipes[len(outs)].recv())

        return outs


# if __name__ == '__main__':
#     q = mp.Queue(maxsize=NCORE)
#     iolock = mp.Lock()
#     pool = mp.Pool(NCORE, initializer=process, initargs=(q, iolock))
#     for stuff in range(20):
#         q.put(stuff)  # blocks until q below its max size
#         with iolock:
#             print("queued", stuff)
#     for _ in range(NCORE):  # tell workers we're done
#         q.put(None)
#     pool.close()
#     pool.join()


# def index_cumulative(index):
#     if alpha == 1:
#         return index + 1
#     return (alpha ** (1 + index) - 1) / (alpha - 1)
#
#
# alpha = 1.01
# N = 10  # len(memory)
# total = index_cumulative(N - 1)
#
#
# def get_range():
#     now = index_cumulative(i)
#     prev = index_cumulative(i - 1)
#
#     dist_range = [prev / total, now / total]
#
#     print(f'range for index {i}: {dist_range}, span={(now - prev) / total}')
#
#     return dist_range
#
#
# # for i in range(N):
# #     get_range()
#
#
# # Now, get index given value in dist range
# def get_index(value):
#     assert 0 <= value <= 1, 'Expected probability.'
#
#     now = value * total
#     alpha_exponential = now * (alpha - 1) + 1
#     unrounded_index = math.log(alpha_exponential, alpha) - 1
#     index = max(0, round(unrounded_index))
#
#     print(f'index for value {value}: {index} (rounded), {unrounded_index} (un-rounded)')
#
#     return index


alpha = 1.01
N = 10  # len(memory)


def index_cumulative(index):
    return index + 1 if alpha == 1 \
        else (alpha ** (1 + index) - 1) / (alpha - 1)


total = index_cumulative(N - 1)


def prioritized_index(rand):  # rand is a random value between 0 and 1.
    return max(0, round(math.log(rand * total * (alpha - 1) + 1, alpha) - 1))


def proba(index):  # Proba of sampling that index
    return (index_cumulative(index) - index_cumulative(index - 1)) / total


for r in [0, 0.05, 0.1, 0.2, 0.3, 1]:
    i = prioritized_index(r)
    print(f'index for random float {r}: {i}, proba of index: {proba(i)}')  # O(1) read/sample complexity


# Compared to https://stackoverflow.com/a/61064196/22002059
# But what did PER use besides sumtree?
# https://docs.python.org/3/library/bisect.html
# Insert into list = sorted(list) via bisect.insort(list, value, key=lambda x: x.)
# Better: https://grantjenks.com/docs/sortedcontainers/ instead of sorted

# Insert into list = SortedList(list) via list.bisect(value, key=lambda x: x.)

# Or: https://pypi.org/project/blist/

# >>> from blist import sortedlist
# >>> my_list = sortedlist([3,7,2,1])
# >>> my_list
# sortedlist([1, 2, 3, 7])
# >>> my_list.add(5)
# >>> my_list[3]
# 5
# >>>
# The sortedlist constructor takes an optional “key” argument, which may be used to change the sort order just like the sorted() function.
# http://stutzbachenterprises.com/blist/sortedlist.html

# logn insert, 1 sample (really fast for offline, still fast online)

# No, bisect/list might not be good  https://stackoverflow.com/a/53023435/22002059
# You can find the insertion point in O(log n) time, but the insertion step that follows is O(n), making this a rather expensive way to sort.
#
# If you are using this to sort m elements, you have a O(m^2) (quadratic) solution for what should only take O(m log m) time with TimSort (the sorting algorithm used by the sorted() function).
#  Can use https://grantjenks.com/docs/sortedcontainers/

# According to PER about sumtree: "allowing O(log N ) updates and sampling" - B.2.1
# An embarrassingly simplified algorithm for prioritized experience replay that runs faster

# bisect can do this already!!!!
# https://stackoverflow.com/a/30657196/22002059 for inserting at index in O(log(N))
# Search algorithm for finding index to insert at - note: insert takes O(n) unless using blist?
# mid = (s_idx + e_idx) / 2
# if e_idx < s_idx:
#     return s_idx
# if self.event_list[mid][0] >= interval[1]:
#     return self.find(s_idx, mid - 1, interval)
# elif self.event_list[mid][1] <= interval[0]:
#     return self.find(mid + 1, e_idx, interval)
# else:
#     return -1


# Oh maybe better: https://stackoverflow.com/a/30657878/22002059 but apache2.0
# Would have to include apache license as well ?


# bisect + skiplist - how to combine? oh no need. assumes sorted
# https://gist.github.com/sachinnair90/3bee2ef7dd3ff0dc5aec44ec40e2d127
# https://pypi.org/project/pyskiplist/

# Solution: Just this:
# # https://pypi.org/project/pyskiplist/

# https://pyskiplist.readthedocs.io/en/stable/
# insert(key, value)
# Insert a key-value pair in the list.
# Also, replace, and delete!

# The pair is inserted at the correct location so that the list remains sorted on key.
# If a pair with the same key is already in the list, then the pair is appended after all other pairs with that key.
