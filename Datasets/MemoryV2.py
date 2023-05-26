# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import numpy as np

import torch
import torch.multiprocessing as mp


class Mem:
    def __init__(self, mem, path=None):
        self.path = path
        self.mem = torch.as_tensor(mem)
        self.is_mmap = False

        self.shape = self.mem.shape
        _, self.dtype = str(self.mem.dtype).split('.')

    def __getitem__(self, ind):
        assert self.shape

        if self.is_mmap:
            mem = np.memmap(self.path, self.dtype, 'r+', shape=self.shape)

            return mem[ind]

        return self.mem[ind]

    def __setitem__(self, ind, value):
        assert self.shape

        if self.is_mmap:
            mem = np.memmap(self.path, self.dtype, 'r+', shape=self.shape)
            mem[ind] = value
            mem.flush()  # Write to hard disk
        else:
            self.mem[ind] = value

    def tensor(self):
        mem = np.memmap(self.path, self.dtype, 'r+', shape=self.shape) if self.is_mmap \
            else self.mem

        return torch.as_tensor(mem)

    def gpu(self):
        self.mem = self.tensor().cuda().share_memory_()
        self.is_mmap = False  # TODO Update in multiproc?

        return self

    def shared(self):
        self.mem = self.tensor().cpu().share_memory_()
        self.is_mmap = False  # TODO Update in multiproc?

        return self

    def mmap(self):
        if not self.is_mmap:
            mmap_file = np.memmap(self.path, self.dtype, 'w+', shape=self.shape)
            if self.shape:
                mmap_file[:] = self.mem[:]
            else:
                mmap_file[...] = self.mem  # In case of 0-dim array
            mmap_file.flush()  # Write to hard disk
            self.mem = None  # TODO Update in multiproc?
            self.is_mmap = True  # TODO Update in multiproc?

        return self

    def __bool__(self):
        return bool(np.memmap(self.path, self.dtype, 'r+', shape=self.shape) if self.is_mmap else self.mem)

    def __len__(self):
        return self.shape[0]


class Memory:
    def __init__(self, device=None, device_capacity=None, ram_capacity=None, cache_capacity=None, hd_capacity=None):
        self.path = './ReplayBuffer'  # /DatasetUniqueIdentifier
        self.path += '/Test'

        manager = mp.Manager()

        self.index = manager.list()
        self.episode_batches = manager.list([()])

    def add(self, batch):
        episode_batches_ind = len(self.episode_batches) - 1
        step = len(self.episode_batches[-1])

        for key in batch:
            batch[key] = Mem(batch[key], f'{self.path}/{episode_batches_ind}_{step}_{key}')  # .to(non_blocking=True)
            batch[key].shared()

        batch_size = 1

        for mem in batch.values():
            if mem.shape and len(mem) > 1:
                batch_size = len(mem)
                break

        self.index.extend(enumerate([episode_batches_ind] * batch_size))

        self.episode_batches[-1] = self.episode_batches[-1] + (batch,)

        if batch['done']:
            self.episode_batches.append(())

    def episode(self, ind):
        ind, episode_batches_ind = self.index[ind]
        batches = self.episode_batches[episode_batches_ind]

        return Episode(batches, ind)

    def __getitem__(self, ind):
        return self.episode(ind)

    def __len__(self):
        return len(self.index)


class Episode:
    def __init__(self, batches, ind):
        self.batches = batches
        self.ind = ind

    def datum(self, key):
        return Experience(self.batches, key, self.ind)

    def __getitem__(self, key):
        return self.datum(key)

    def __setitem__(self, key, value):
        for batch in self.batches:
            batch[key][self.ind] = value

    def __len__(self):
        return len(self.batches)


class Experience:
    def __init__(self, batches, key, ind):
        self.batches = batches
        self.key = key
        self.ind = ind

    def step(self, step):
        return self.batches[step][self.key][self.ind]

    def __getitem__(self, step):
        return self.step(step)

    def __setitem__(self, step, experience):
        self.batches[step][self.key][self.ind] = experience

    def len(self):
        return len(self.batches)


def f1(m):
    while True:
        _start = time.time()
        print(m.episode(0)['hi'][0][0, 0, 0].item(), time.time() - _start, 'get2', len(m))
        time.sleep(3)


def f2(m):
    while True:
        _start = time.time()
        print(m.episode(0)['hi'][0][0, 0, 0].item(), time.time() - _start, 'get2', len(m))
        time.sleep(3)


if __name__ == '__main__':
    M = Memory()
    adds = 0
    for _ in range(2):
        for _ in range(128 - 1):
            d = {'hi': np.ones([2560, 3, 32, 32]), 'done': False}
            start = time.time()
            M.add(d)
            adds += time.time() - start
        done = {'hi': np.ones([2560, 3, 32, 32]), 'done': True}
        start = time.time()
        M.add(done)
        adds += time.time() - start
    print(adds, 'adds')
    p1 = mp.Process(name='p1', target=f1, args=(M,))
    p2 = mp.Process(name='p', target=f2, args=(M,))
    p1.start()
    p2.start()
    start = time.time()
    e = M.episode(0)
    e['hi'][0][0] = 5
    print(time.time() - start, 'set')
    p1.join()
    p2.join()
