# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import atexit
import os
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

import torch
import torch.multiprocessing as mp


class Memory:
    def __init__(self, device=None, cache_capacity=None, gpu_capacity=None, ram_capacity=None, hd_capacity=None):
        self.path = './ReplayBuffer'  # /DatasetUniqueIdentifier
        self.path += '/Test'

        self.num_batches = 0

        manager = mp.Manager()

        self.batches = manager.list()
        self.in_episode_batches = []
        self.episodes = []

        atexit.register(self.cleanup)

    def update(self):  # Maybe truly-shared list variable can tell workers when to do this
        for batch in self.batches[self.num_batches:]:
            batch_size = 1

            for mem in batch.values():
                if mem.shape and len(mem) > 1:
                    batch_size = len(mem)
                    break

            self.in_episode_batches.append(batch)
            self.episodes.extend([Episode(self.in_episode_batches, i) for i in range(batch_size)])

            if batch['done']:
                self.in_episode_batches = []

            self.num_batches += 1

    def add(self, batch):
        batch = Batch({key: Mem(batch[key], f'{self.path}/{self.num_batches}_{key}').shared()
                       for key in batch})
        self.batches.append(batch)
        self.update()  # Each worker must update

    def episode(self, ind):
        return self.episodes[ind]

    def __getitem__(self, ind):
        return self.episode(ind)

    def __len__(self):
        return len(self.episodes)

    def cleanup(self):
        for batch in self.batches:
            for mem in batch.values():
                if mem.mode == 'shared':
                    mem.mem.close()
                    mem.mem.unlink()


class Episode:
    def __init__(self, in_episode_batches, ind):
        self.in_episode_batches = in_episode_batches
        self.ind = ind

    def step(self, step):
        return Experience(self.in_episode_batches, step, self.ind)

    def __getitem__(self, step):
        return self.step(step)

    def __len__(self):
        return len(self.in_episode_batches)

    def __iter__(self):
        return (self.step(i) for i in range(len(self)))


class Experience:
    def __init__(self, in_episode_batches, step, ind):
        self.in_episode_batches = in_episode_batches
        self.step = step
        self.ind = ind

    def datum(self, key):
        return self.in_episode_batches[self.step][key][self.ind]

    def keys(self):
        return self.in_episode_batches[self.step].keys()

    def values(self):
        return [self.datum(key) for key in self.keys()]

    def items(self):
        return zip(self.keys(), self.values())

    def __getitem__(self, key):
        return self.datum(key)

    def __setitem__(self, key, experience):
        self.in_episode_batches[self.step][key][self.ind] = experience

    def __iter__(self):
        return iter(self.in_episode_batches[self.step].keys())


class Batch(dict):
    def __init__(self, _dict, **kwargs):
        super().__init__()
        self.__dict__ = self
        self.update({**_dict, **kwargs})


class Mem:
    def __init__(self, mem, path=None):
        self.path = path
        self.mem = np.array(mem)
        self.mode = 'tensor'

        self.shape = self.mem.shape
        self.dtype = self.mem.dtype

        self.main_worker = os.getpid()

    def get(self):
        if self.mode == 'mmap':
            while True:  # Non-main workers wait to sync
                try:
                    return np.memmap(self.path, self.dtype, 'r+', shape=self.shape)
                except FileNotFoundError as e:
                    if self.main_worker == os.getpid():
                        raise e
                    continue
        elif self.mode == 'shared':
            return np.ndarray(self.shape, dtype=self.dtype, buffer=self.mem.buf)
        else:
            return self.mem

    def __getitem__(self, ind):
        assert self.shape
        mem = self.get()[ind]

        if self.mode == 'shared':
            # Note: Nested sets won't work
            return mem.copy()

        return mem

    def __setitem__(self, ind, value):
        assert self.shape

        if self.mode == 'mmap':
            mem = self.get()
            mem[ind] = value
            mem.flush()  # Write to hard disk
        elif self.mode == 'shared':
            self.get()[ind] = value
        else:
            self.mem[ind] = value

    def tensor(self):
        return torch.as_tensor(self.get()).to(non_blocking=True)

    def gpu(self):
        self.mem = self.tensor().cuda().share_memory_().to(non_blocking=True)
        self.mode = 'gpu'

        return self

    def shared(self):
        if self.mode != 'shared':
            name = '_'.join(self.path.rsplit('/', 2)[1:]) + '_' + str(id(self))
            mem = self.tensor().numpy()
            link = SharedMemory(create=True, name=name,  size=mem.nbytes)
            mem_ = np.ndarray(self.shape, dtype=self.dtype, buffer=link.buf)
            if self.shape:
                mem_[:] = mem[:]
            else:
                mem_[...] = mem  # In case of 0-dim array

            self.mem = link
            self.mode = 'shared'

        return self

    def mmap(self):  # TODO Create context for cleaning up shared memory
        if self.mode != 'mmap':
            if self.main_worker == os.getpid():
                mmap_file = np.memmap(self.path, self.dtype, 'w+', shape=self.shape)
                if self.shape:
                    mmap_file[:] = self.get()[:]
                else:
                    mmap_file[...] = self.get()  # In case of 0-dim array
                mmap_file.flush()  # Write to hard disk

            self.mem = None
            self.mode = 'mmap'

        return self

    def __bool__(self):
        return bool(self.get())

    def __len__(self):
        return self.shape[0]


def f1(m):
    while True:
        _start = time.time()
        m.update()
        print(m.episode(0)[0]['hi'][0, 0, 0].item(), time.time() - _start, 'get1', m.num_batches)
        time.sleep(3)


def f2(m):
    while True:
        _start = time.time()
        m.update()
        print(m.episode(-1)[0]['hi'][0, 0, 0].item(), time.time() - _start, 'get2', m.num_batches)
        time.sleep(3)


if __name__ == '__main__':
    M = Memory()

    adds = 0
    episodes, steps = 64, 5
    for _ in range(episodes):
        for _ in range(steps - 1):
            d = {'hi': np.random.rand(256, 3, 32, 32), 'done': False}  # Batches
            start = time.time()
            M.add(d)
            adds += time.time() - start
        d = {'hi': np.random.rand(256, 3, 32, 32), 'done': True}  # Last batch
        start = time.time()
        M.add(d)
        adds += time.time() - start
    print(adds, 'adds')

    start = time.time()
    M.episode(0).step(0)['hi'] = 5
    print(time.time() - start, 'set')

    p1 = mp.Process(name='p1', target=f1, args=(M,))
    p2 = mp.Process(name='p2', target=f2, args=(M,))
    p1.start()
    p2.start()

    adds = 0
    episodes, steps = 1, 5
    for _ in range(episodes):
        for _ in range(steps - 1):
            d = {'hi': np.random.rand(256, 3, 32, 32), 'done': False}  # Batches
            start = time.time()
            M.add(d)
            adds += time.time() - start
        d = {'hi': np.random.rand(256, 3, 32, 32), 'done': True}  # Last batch
        start = time.time()
        M.add(d)
        adds += time.time() - start
    print(adds, 'adds another')

    start = time.time()
    M.episode(-1).step(0)['hi'] = 5
    print(time.time() - start, 'set another')
    p1.join()
    p2.join()
