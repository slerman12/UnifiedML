# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from math import inf
import os
import time

import numpy as np

import torch
import torch.multiprocessing as mp


class Memory:
    def __init__(self, save_path='./ReplayBuffer/Test', num_workers=1, gpu_capacity=0, ram_capacity=1e6, hd_capacity=inf):
        self.gpu_capacity = gpu_capacity
        self.ram_capacity = ram_capacity
        self.hd_capacity = hd_capacity

        self.id = id(self)
        self.worker = 0
        self.main_worker = os.getpid()

        self.path = save_path

        manager = mp.Manager()

        self.batches = manager.list()
        self.episode_trace = []
        self.episodes = []

        # Rewrite tape
        self.queues = [Queue()] + [mp.Queue() for _ in range(num_workers - 1)]

        # Counters
        self.num_batches_deleted = torch.zeros([], dtype=torch.int64).share_memory_()
        self.num_batches = self.num_experiences = self.num_experiences_mmapped = self.num_episodes_deleted = 0

    def rewrite(self):  # TODO Thread w sync?
        # Before enforce_capacity changes index
        while not self.queue.empty():
            experience, episode, step = self.queue.get()

            for key in experience:
                self.episode(episode)[step][key] = experience

    def update(self):  # Maybe truly-shared list variable can tell workers when to do this  TODO Thread
        num_batches_deleted = self.num_batches_deleted.item()
        self.num_batches = max(self.num_batches, num_batches_deleted)

        for batch in self.batches[self.num_batches - num_batches_deleted:]:
            batch_size = batch.size()

            if not self.episode_trace:
                self.episodes.extend([Episode(self.episode_trace, i) for i in range(batch_size)])

            self.episode_trace.append(batch)

            self.num_batches += 1

            if batch['done']:
                self.episode_trace = []

            self.num_experiences += batch_size
            self.enforce_capacity()  # Note: Last batch does enter RAM before capacity is enforced

    def add(self, batch):  # TODO Should be its own thread  https://stackoverflow.com/questions/14234547/threads-with-decorators
        assert self.main_worker == os.getpid(), 'Only main worker can send new batches.'

        batch_size = Batch(batch).size()

        gpu = self.num_experiences + batch_size <= self.gpu_capacity
        shared = self.num_experiences + batch_size <= self.gpu_capacity + self.ram_capacity
        mmap = self.num_experiences + batch_size <= self.gpu_capacity + self.ram_capacity + self.hd_capacity

        mode = 'gpu' if gpu else 'shared' if shared else 'mmap' if mmap \
            else next(iter(self.episodes[0].batch(0).values())).mode  # Oldest batch

        batch = Batch({key: Mem(batch[key], f'{self.path}/{self.num_batches}_{key}_{self.id}').to(mode)
                       for key in batch})

        self.batches.append(batch)
        self.update()

    def writable_tape(self, batch, ind, step):  # TODO Should be its own thread
        assert self.main_worker == os.getpid(), 'Only main worker can send rewrites across the memory tape.'

        for batch, ind, step in zip(batch, ind, step):
            self.queues[int(ind % self.worker)].put((batch, ind, step))

        self.rewrite()

    def enforce_capacity(self):
        while self.num_experiences > self.gpu_capacity + self.ram_capacity + self.hd_capacity:
            batch = self.episodes[0].batch(0)
            batch_size = batch.size()

            self.num_experiences -= batch_size

            if self.main_worker == os.getpid():
                self.num_batches_deleted[...] = self.num_batches_deleted + 1
                del self.batches[0]
                for i, mem in enumerate(batch.values()):
                    mem.delete()  # Delete oldest batch

            if next(iter(batch.values())).mode == 'mmap':
                self.num_experiences_mmapped -= batch_size

            del self.episodes[0][0]
            if not len(self.episodes[0]):
                del self.episodes[:batch_size]
                self.num_episodes_deleted += batch_size  # getitem ind = mem.index - self.num_episodes_deleted

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
    def __init__(self, episode_trace, ind):
        self.episode_trace = episode_trace
        self.ind = ind

    def batch(self, step):
        return self.episode_trace[step]

    def experience(self, step):
        return Experience(self.episode_trace, step, self.ind)

    def __getitem__(self, step):
        return self.experience(step)

    def __len__(self):
        return len(self.episode_trace)

    def __iter__(self):
        return (self.experience(i) for i in range(len(self)))

    def __delitem__(self, ind):
        self.episode_trace.pop(ind)


class Experience:
    def __init__(self, episode_trace, step, ind):
        self.episode_trace = episode_trace
        self.step = step
        self.ind = ind

    def datum(self, key):
        return self.episode_trace[self.step][key][self.ind]

    def keys(self):
        return self.episode_trace[self.step].keys()

    def values(self):
        return [self.datum(key) for key in self.keys()]

    def items(self):
        return zip(self.keys(), self.values())

    def __getitem__(self, key):
        return self.datum(key)

    def __getattr__(self, key):
        return self.datum(key)

    def __setitem__(self, key, experience):
        self.episode_trace[self.step][key][self.ind] = experience

    def __iter__(self):
        return iter(self.episode_trace[self.step].keys())


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
        elif self.mode == 'shared':
            return self.mem
        else:
            return self.mem

    def __getitem__(self, ind):
        assert self.shape
        return self.get()[ind]

    def __setitem__(self, ind, value):
        assert self.shape

        if self.mode == 'mmap':
            mem = self.get()
            mem[ind] = value
            mem.flush()  # Write to hard disk
        elif self.mode == 'shared':
            self.get()[ind] = value
        else:
            self.get()[ind] = value

    def tensor(self):
        return torch.as_tensor(self.get()).to(non_blocking=True)

    def gpu(self):
        if self.mode != 'gpu':
            self.mem = torch.as_tensor(self.get()).cuda().to(non_blocking=True)
        self.mode = 'gpu'

        return self

    def shared(self):  # Would pinned memory be better? tensor.pin_memory()?  https://pytorch.org/docs/stable/data.html
        if self.mode != 'shared':
            self.mem = torch.as_tensor(self.get()).share_memory_().to(non_blocking=False).pin_memory()
        self.mode = 'shared'

        return self

    def mmap(self):
        if self.mode != 'mmap':
            if self.main_worker == os.getpid():  # For online transitions
                mem = self.get()
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

    def __bool__(self):
        return bool(self.get())

    def __len__(self):
        return self.shape[0]

    def delete(self):
        if self.mode == 'mmap':
            os.remove(self.path)


def offline(m):
    while True:
        _start = time.time()
        # m.update()
        print(m.episode(-1)[-1].hi[0, 0, 0].item(), time.time() - _start, 'offline')
        time.sleep(3)


# Note Version10 is faster for Online
def online(m):
    while True:
        _start = time.time()
        m.update()
        print(m.episode(-1)[-1].hi[0, 0, 0].item(), time.time() - _start, 'online')
        time.sleep(3)


import gc
import sys


# https://stackoverflow.com/a/53705610
# https://stackoverflow.com/questions/54361763/pytorch-why-is-the-memory-occupied-by-the-tensor-variable-so-small
def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


if __name__ == '__main__':
    M = Memory(num_workers=3)

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
    M.episode(-1).experience(-1)['hi'] = 5
    print(time.time() - start, 'set')

    p1 = mp.Process(name='offline', target=offline, args=(M,))
    p2 = mp.Process(name='online', target=online, args=(M,))
    p1.start()
    p2.start()
    time.sleep(3)  # Online hd_capacity requires a moment! (Before any additional updates) (for mp to copy/spawn)

    adds = 0
    episodes, steps = 1, 5
    for _ in range(episodes):
        for _ in range(steps - 1):
            d = {'hi': np.random.rand(256, 3, 32, 32), 'done': False}  # Batches
            start = time.time()
            M.add(d)
            adds += time.time() - start
            # setattr(M.episode(0).batch(0), 'd', d)
            # print(get_obj_size(M.episode(0).batch(0)))
        d = {'hi': np.random.rand(256, 3, 32, 32), 'done': True}  # Last batch
        start = time.time()
        M.add(d)
        adds += time.time() - start
    print(adds, 'adds another')

    start = time.time()
    M.episode(-1).experience(-1)['hi'] = 5
    print(time.time() - start, 'set another')

    import random

    i = 0

    while True:
        for _ in range(random.randint(0, 5)):
            d = {'hi': np.full([256, 3, 32, 32], i), 'done': False}  # Batches
            start = time.time()
            M.add(d)
            print(time.time() - start, 'add another')
            # M.episode(-1).experience(-1)['hi'] = 7
            time.sleep(3)
            i += 1
        d = {'hi': np.full([256, 3, 32, 32], i), 'done': True}  # Last batch
        start = time.time()
        M.add(d)
        print(time.time() - start, 'add another')
        # M.episode(-1).experience(-1)['hi'] = 7
        time.sleep(3)
        i += 1

    p1.join()
    p2.join()
