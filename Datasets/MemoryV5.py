# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from math import inf
import atexit
import contextlib
import os
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

import torch
import torch.multiprocessing as mp


class Memory:
    def __init__(self, save_path='./ReplayBuffer/Test', num_workers=1, gpu_capacity=0, ram_capacity=1000000,
                 hd_capacity=inf):
        self.worker = 0
        self.main_worker = os.getpid()

        self.path = save_path  # +/DatasetUniqueIdentifier

        # Counters
        self.num_batches = self.num_experiences = self.num_experiences_mmapped = \
            self.num_batches_deleted = self.num_episodes_deleted = 0

        # GPU or CPU RAM, and hard disk
        self.gpu_capacity = gpu_capacity
        self.ram_capacity = ram_capacity
        self.hd_capacity = hd_capacity

        # Non-redundant mmap-ing
        self.last_mmapped_ind = (0, 0)

        manager = mp.Manager()

        self.batches = manager.list()
        self.in_episode_batches = []  # Episode trace
        self.episodes = []

        # Rewrite tape
        self.queues = [Queue()] + [mp.Queue() for _ in range(num_workers - 1)]

        atexit.register(self.cleanup)

    def update(self, rewrite=True, add=True):  # Maybe truly-shared list variable can tell workers when to do this
        if rewrite:
            while not self.queue.empty():
                experience, episode, step = self.queue.get()

                for key in experience:
                    self.episode(episode)[step][key] = experience

        if add:
            for batch in self.batches[self.num_batches:]:
                batch_size = batch.size()

                self.in_episode_batches.append(batch)
                self.episodes.extend([Episode(self.in_episode_batches, i) for i in range(batch_size)])

                if batch['done']:
                    self.in_episode_batches = []

                self.num_batches += 1

                if self.main_worker != os.getpid():
                    self.num_experiences += batch_size
                    self.enforce_capacity()

    def add(self, batch, ind=None, step=None):
        assert not self.worker

        if ind is None or step is None:
            batch_size = 1

            for mem in batch.values():
                if mem.shape and len(mem) > 1:
                    batch_size = len(mem)
                    break

            self.num_experiences += batch_size
            self.enforce_capacity()

            # Shared GPU or CPU RAM, depending on capacity
            mode = 'gpu' if self.num_experiences < self.gpu_capacity else 'shared'
            absolute_num_batches = self.num_batches + self.num_batches_deleted
            batch = Batch({key: Mem(batch[key], f'{self.path}/{absolute_num_batches}_{key}').to(mode)
                           for key in batch})

            self.batches.append(batch)
            self.update()  # Each worker must update
        else:
            # Writable tape
            for batch, ind, step in zip(batch, ind, step):
                self.queues[int(ind % self.worker)].put((batch, ind, step))

    def enforce_capacity(self):
        # Delete oldest batch
        while self.num_experiences > self.gpu_capacity + self.ram_capacity + self.hd_capacity:
            batch_size = self.episodes[0][0].size()

            self.num_experiences -= batch_size
            self.num_batches -= 1
            self.num_batches_deleted += 1

            if self.main_worker == os.getpid():
                del self.batches[0]
                for mem in self.episodes[0][0].values():
                    mem.delete()  # TODO

            del self.episodes[0][0]
            if not len(self.episodes[0]):
                del self.episodes[:batch_size]
                self.num_episodes_deleted += batch_size  # getitem ind = mem.index - self.num_episodes_deleted

        # MMAP oldest batch
        while self.num_experiences - self.num_experiences_mmapped > self.gpu_capacity + self.ram_capacity:
            episode_ind, step = self.last_mmapped_ind

            batch = self.episode(episode_ind)[step]
            batch_size = batch.size()
            while self.num_experiences - self.num_experiences_mmapped > self.gpu_capacity + self.ram_capacity:
                for mem in batch.values():
                    mem.mmap()
                self.num_experiences_mmapped += batch_size

            if episode_ind < len(self.episode(episode_ind)) - 1:
                self.last_mmapped_ind[1] += 1
            else:
                self.last_mmapped_ind = (episode_ind + batch_size, 0)

    def load(self):
        pass

    def save(self):
        pass

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

    def set_worker(self, worker):
        self.worker = worker

    @property
    def queue(self):
        return self.queues[self.worker]


class Queue:
    def __init__(self):
        self.queue = []

    def get(self):
        return self.queue.pop()

    def put(self, item):
        self.queue.append(item)

    def empty(self):
        return not len(self.queue)


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

    def __delitem__(self, ind):
        self.in_episode_batches.pop(ind)


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
    def __init__(self, _dict=None, **kwargs):
        super().__init__()
        self.__dict__ = self  # Allows access via attributes
        self.update({**(_dict or {}), **kwargs})

    def size(self):
        for mem in self.values():
            if hasattr(mem, '__len__') and len(mem) > 1:
                return len(mem)

        return 1


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
            while True:  # Online syncing
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
        if self.mode != 'gpu':
            with self.cleanup() as mem:
                self.mem = torch.as_tensor(mem).cuda().to(non_blocking=True)
            self.mode = 'gpu'

        return self

    def shared(self):
        if self.mode != 'shared':
            with self.cleanup() as mem:
                name = '_'.join(self.path.rsplit('/', 2)[1:]) + '_' + str(id(self))
                if isinstance(mem, torch.Tensor):
                    mem = mem.numpy()
                link = SharedMemory(create=True, name=name,  size=mem.nbytes)
                mem_ = np.ndarray(self.shape, dtype=self.dtype, buffer=link.buf)
                if self.shape:
                    mem_[:] = mem[:]
                else:
                    mem_[...] = mem  # In case of 0-dim array

            self.mem = link
            self.mode = 'shared'

        return self

    def mmap(self):
        if self.mode != 'mmap':
            if self.main_worker == os.getpid():  # For online transitions
                with self.cleanup() as mem:
                    mmap_file = np.memmap(self.path, self.dtype, 'w+', shape=self.shape)
                    if self.shape:
                        mmap_file[:] = mem[:]
                    else:
                        mmap_file[...] = mem  # In case of 0-dim array
                    mmap_file.flush()  # Write to hard disk

            self.mem = None
            self.mode = 'mmap'

        return self

    def to(self, mode):
        if mode == 'gpu':
            return self.gpu()
        elif mode == 'shared':
            return self.shared()
        elif mode == 'mmap':
            return self.mmap()
        else:
            assert False, f'Mode "{mode}" not supported."'

    @contextlib.contextmanager
    def cleanup(self):
        mem = self.get()
        yield mem
        if self.mode == 'shared':
            mem.close()
            mem.unlink()

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
