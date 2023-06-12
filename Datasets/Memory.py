# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from math import inf
import atexit
import contextlib
import os
import warnings
from multiprocessing.shared_memory import SharedMemory
import resource
from pathlib import Path

import numpy as np

import torch
import torch.multiprocessing as mp


class Memory:
    def __init__(self, save_path='./ReplayBuffer/Test', num_workers=1, gpu_capacity=0, pinned_capacity=0,
                 tensor_ram_capacity=0, ram_capacity=1e6, hd_capacity=inf):
        self.gpu_capacity = gpu_capacity
        self.pinned_capacity = pinned_capacity
        self.tensor_ram_capacity = tensor_ram_capacity
        self.ram_capacity = ram_capacity
        self.hd_capacity = hd_capacity

        self.id = id(self)
        self.worker = 0
        self.main_worker = os.getpid()

        self.save_path = save_path

        manager = mp.Manager()

        self.batches = manager.list()
        self.episode_trace = []
        self.episodes = []

        # Rewrite tape
        self.queues = [Queue()] + [mp.Queue() for _ in range(num_workers - 1)]

        # Counters
        self.num_batches_deleted = torch.zeros([], dtype=torch.int64).share_memory_()
        self.num_batches = self.num_experiences = self.num_experiences_mmapped = self.num_episodes_deleted = 0

        atexit.register(self.cleanup)

        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)  # Shared memory can create a lot of file descriptors
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))  # Increase soft limit to hard limit

    def rewrite(self):  # TODO Thread w sync?
        # Before enforce_capacity changes index
        while not self.queue.empty():
            experience, episode, step = self.queue.get()

            for key in experience:
                self.episode(episode)[step][key] = experience

    def update(self):  # Maybe truly-shared list variable can tell workers when to do this / lock  TODO Thread
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

    def add(self, batch):  # TODO Be own thread https://stackoverflow.com/questions/14234547/threads-with-decorators
        assert self.main_worker == os.getpid(), 'Only main worker can send new batches.'

        batch_size = Batch(batch).size()

        capacities = [inf if capacity == 'inf' else capacity for capacity in [self.gpu_capacity, self.pinned_capacity,
                                                                              self.tensor_ram_capacity,
                                                                              self.ram_capacity, self.hd_capacity]]
        gpu = self.num_experiences + batch_size <= sum(capacities[:1])
        pinned = self.num_experiences + batch_size <= sum(capacities[:2])
        shared_tensor = self.num_experiences + batch_size <= sum(capacities[:3])
        shared = self.num_experiences + batch_size <= sum(capacities[:4])
        mmap = self.num_experiences + batch_size <= sum(capacities[:5])

        mode = 'gpu' if gpu else 'pinned' if pinned else 'shared_tensor' if shared_tensor \
            else 'shared' if shared else 'mmap' if mmap \
            else next(iter(self.episodes[0].batch(0).values())).mode  # Oldest batch

        batch = Batch({key: Mem(batch[key], f'{self.save_path}/{self.num_batches}_{key}_{self.id}').to(mode)
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

    def trace(self, ind):
        return self.episodes[ind][0].episode_trace

    @property
    def traces(self):
        yield from (self.trace(i) for i in range(len(self.episodes)))

    def episode(self, ind):
        return self.episodes[ind]

    def __getitem__(self, ind):
        return self.episode(ind)

    def __len__(self):
        return len(self.episodes)

    def cleanup(self):
        for batch in self.batches:
            for mem in batch.mems:
                if mem.mode == 'shared':
                    mem.shm.close()
                    mem.shm.unlink()

    def set_worker(self, worker):
        self.worker = worker

    @property
    def queue(self):
        return self.queues[self.worker]

    def load(self, load_path=None):
        assert self.main_worker == os.getpid(), 'Only main worker can call load.'

        if load_path is None:
            load_path = self.save_path

        mmap_paths = sorted(Path(load_path).glob('*'))
        batch = {}
        previous_num_batches = 0

        for i, mmap_path in enumerate(mmap_paths):
            _, num_batches, key, identifier = mmap_path.stem.rsplit('_', 3)

            if i == 0:
                self.id = identifier
                self.num_batches_deleted[...] = self.num_batches
            else:
                if self.id != identifier:
                    warnings.warn(f'Found Mems with multiple identifiers in load path {load_path}. Using id={self.id}.')
                    continue

                self.num_batches = num_batches

                if self.num_batches > previous_num_batches:
                    self.add(batch)
                    batch = {}

            batch[key] = Mem(None, path=mmap_path).load().mem

            previous_num_batches = self.num_batches

    def save(self):
        assert self.main_worker == os.getpid(), 'Only main worker can call save.'

        for batch in self.traces:
            for mem in batch.mems:
                mem.save()


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

    @property
    def mems(self):  # An element can be Mem or datums
        yield from self.values()

    def size(self):
        for mem in self.values():
            if hasattr(mem, '__len__') and len(mem) > 1:
                return len(mem)

        return 1


class Mem:
    def __init__(self, mem, path=None):
        self.shm = None
        self.mem = None if mem is None else np.array(mem)
        self.path = path
        self.saved = False

        self.mode = None if mem is None else 'ndarray'

        if mem is None:
            self.shape, self.dtype = (), None
        else:
            self.shape = self.mem.shape
            self.dtype = self.mem.dtype
            self.path += '_' + str(tuple(self.shape)) + '_' + self.dtype.name

        self.name = '_'.join(self.path.rsplit('/', 4)[1:])

        self.main_worker = os.getpid()

    def __getstate__(self):
        if self.mode == 'shared':
            self.shm.close()
        return self.path, self.saved, self.mode, self.main_worker, self.shape, self.dtype, \
            *((self.mem,) if self.mode in ('pinned', 'shared_tensor') else ())

    def __setstate__(self, state):
        self.path, self.saved, self.mode, self.main_worker, self.shape, self.dtype, *mem = state
        self.name = '_'.join(self.path.rsplit('/', 4)[1:])

        if self.mode == 'shared':
            self.shm = SharedMemory(name=self.name)
            self.mem = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        elif self.mode == 'mmap':
            self.shm = None
            self.mem = np.memmap(self.path, self.dtype, 'r+', shape=self.shape)
        else:
            self.shm = None
            self.mem = mem

    def __getitem__(self, ind):
        assert self.shape
        return self.mem[ind]

    def __setitem__(self, ind, value):
        assert self.shape

        self.mem[ind] = value

        if self.mode == 'mmap':
            self.mem.flush()  # Write to hard disk

        self.saved = False

    @property
    def datums(self):
        return self.mem

    def tensor(self):
        return torch.as_tensor(self.mem).to(non_blocking=True)

    def pinned(self):
        if self.mode != 'pinned':
            with self.cleanup():
                self.mem = torch.as_tensor(self.mem).share_memory_().to(non_blocking=True).pin_memory()  # if cuda!
            self.mode = 'pinned'

        return self

    def shared_tensor(self):
        if self.mode != 'shared_tensor':
            with self.cleanup():
                self.mem = torch.as_tensor(self.mem).share_memory_().to(non_blocking=True)
            self.mode = 'shared_tensor'

        return self

    def gpu(self):
        if self.mode != 'gpu':
            with self.cleanup():
                self.mem = torch.as_tensor(self.mem).cuda(non_blocking=True)

            self.mode = 'gpu'

        return self

    def shared(self):
        if self.mode != 'shared':
            with self.cleanup():
                if isinstance(self.mem, torch.Tensor):
                    self.mem = self.mem.numpy()
                mem = self.mem
                self.shm = SharedMemory(create=True, name=self.name, size=mem.nbytes)
                self.mem = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
                if self.shape:
                    self.mem[:] = mem[:]
                else:
                    self.mem[...] = mem  # In case of 0-dim array

            self.mode = 'shared'

        return self

    def mmap(self):
        if self.mode != 'mmap':
            with self.cleanup():
                if self.main_worker == os.getpid():  # For online transitions
                    mem = self.mem
                    self.mem = np.memmap(self.path, self.dtype, 'w+', shape=self.shape)
                    if self.shape:
                        self.mem[:] = mem[:]
                    else:
                        self.mem[...] = mem  # In case of 0-dim array
                    self.mem.flush()  # Write to hard disk
                else:
                    self.mem = np.memmap(self.path, self.dtype, 'r+', shape=self.shape)

            self.mode = 'mmap'
            self.saved = True

        return self

    def to(self, mode):
        if mode == 'pinned':
            return self.pinned()
        if mode == 'shared_tensor':
            return self.shared_tensor()
        if mode == 'shared':
            return self.shared()
        elif mode == 'mmap':
            return self.mmap()
        else:
            assert False, f'Mode "{mode}" not supported."'

    @contextlib.contextmanager
    def cleanup(self):
        yield
        if self.mode == 'shared':
            self.shm.close()
            if self.main_worker == os.getpid():
                self.shm.unlink()
            self.shm = None

    def __bool__(self):
        return bool(self.mem)

    def __len__(self):
        return self.shape[0]

    def load(self):
        if not self.saved:
            _, shape, dtype = self.path.rsplit('_', 2)
            mem = np.memmap(self.path, dtype, 'r+', shape=eval(shape))

            if self.mem is None:
                self.mem = mem
                self.shm = None
                self.mode = 'mmap'
            else:
                if isinstance(self.mem, torch.Tensor):
                    mem = torch.as_tensor(mem)
                if self.shape:
                    self.mem[:] = mem[:]
                else:
                    self.mem[...] = mem  # In case of 0-dim array

            self.saved = True

        return self

    def save(self):
        if not self.saved:
            mmap = np.memmap(self.path, self.dtype, 'w+', shape=self.shape)

            if self.shape:
                mmap[:] = self.mem[:]
            else:
                mmap[...] = self.mem  # In case of 0-dim array

            mmap.flush()  # Write to hard disk
            self.saved = True

    def delete(self):
        with self.cleanup():
            if self.mode == 'mmap':
                if self.main_worker == os.getpid():
                    os.remove(self.path)

        self.saved = False
