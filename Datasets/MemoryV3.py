# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import atexit
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

import torch
import torch.multiprocessing as mp


class Mem:
    # Could make all sub-variables besides path truly shared
    def __init__(self, mem, path=None):
        self.path = path
        # self.mem = np.array(mem)
        self.mem = torch.as_tensor(mem)
        self.mode = 'tensor'  # Must be shared if using cache for sake of mmapping in online

        # These could be tuple elements in self.mem when mmap or shared
        self.shape = self.mem.shape
        # self.dtype = str(self.mem.dtype).split('.')
        _, self.dtype = str(self.mem.dtype).split('.')

        atexit.register(self.cleanup)

    def get(self):
        if self.mode == 'mmap':
            return np.memmap(self.path, self.dtype, 'r+', shape=self.shape)
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

    def gpu(self):  # only can work offline unless workers sync
        self.mem = self.tensor().cuda().share_memory_().to(non_blocking=True)  # TODO Update in multiproc?
        self.mode = 'gpu'  # TODO Update in multiproc?

        return self

    def shared(self):
        # Maybe switch to Ray https://luis-sena.medium.com/sharing-big-numpy-arrays-across-python-processes-abf0dc2a0ab2
        if self.mode != 'shared':
            name = '_'.join(self.path.rsplit('/', 2)[1:]) + '_' + str(id(self))
            mem = self.tensor().numpy()
            link = SharedMemory(create=True, name=name,  size=mem.nbytes)
            mem_ = np.ndarray(self.shape, dtype=self.dtype, buffer=link.buf)
            if self.shape:
                mem_[:] = mem[:]
            else:
                mem_[...] = mem  # In case of 0-dim array

            self.mem = link  # TODO Update in multiproc?
            self.mode = 'shared'  # TODO Update in multiproc?

        return self

    def mmap(self):  # Only one worker needs to mmap; but there be a try catch in the get method for out of sync access
        if self.mode != 'mmap':
            mmap_file = np.memmap(self.path, self.dtype, 'w+', shape=self.shape)
            if self.shape:
                mmap_file[:] = self.get()[:]
            else:
                mmap_file[...] = self.get()  # In case of 0-dim array
            mmap_file.flush()  # Write to hard disk

            self.mem = None  # TODO Update in multiproc?
            self.mode = 'mmap'  # TODO Update in multiproc?

        return self

    def __bool__(self):
        return bool(self.get())

    def __len__(self):
        return self.shape[0]

    def cleanup(self):
        if self.mode == 'shared':  # TODO Should do in between switches too
            self.mem.close()
            self.mem.unlink()


class Memory:
    def __init__(self, device=None, cache_capacity=None, gpu_capacity=None, ram_capacity=None, hd_capacity=None):
        self.path = './ReplayBuffer'  # /DatasetUniqueIdentifier
        self.path += '/Test'

        # Workers can build cached list of Episodes from Manager - when all workers report done, the next add can
        # clear manager

        # Might as well use per-worker pipes? - No
        manager = mp.Manager()  # Serializing takes a long time as episode length; cache every batch locally!

        # Maybe to speed up, can serialize
        # self.batches first step mem can store update-able shared episode len if needed; index can point to [firsts] *
        # batch elements; index can be disabled for online and batches and exps can be sampled
        # self.batches is sequentially ordered w.r.t. steps in episodes
        # Every batch that gets added or indexed (if not key) or set (on every access) is cached
        # self.index = []
        # self.episode_batches = [()]
        self.index = manager.list()
        self.episode_batches = manager.list([()])
        # mem_cache
        # self.cache = {}  # Everything; Maybe a superset with some unshared_ram_capacity saved under Worker1, ...
        # Can also have a finite data_cache e.g. for mmaps and web links and maybe faster than TSM
        # gpu_capacity = 0 if device != cuda

    def add(self, batch):
        done = batch['done']
        episode_batches_ind = len(self.episode_batches) - 1
        step = len(self.episode_batches[-1])

        # Note: Maybe make new dict/Batch non-destructively
        for key in batch:
            batch[key] = Mem(batch[key], f'{self.path}/{episode_batches_ind}_{step}_{key}')
            batch[key].shared()  # Maybe use this only if offline, but then hve to pipe

        batch_size = 1

        for mem in batch.values():
            if mem.shape and len(mem) > 1:
                batch_size = len(mem)
                break

        self.index.extend(enumerate([episode_batches_ind] * batch_size))

        self.episode_batches[-1] = self.episode_batches[-1] + (batch,)

        if done:
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

    def step(self, step):
        return Experience(self.batches, step, self.ind)

    def __getitem__(self, step):
        return self.step(step)

    def __len__(self):
        return len(self.batches)


class Experience:
    def __init__(self, batches, step, ind):
        self.batches = batches
        self.step = step
        self.ind = ind

    def datum(self, key):
        return self.batches[self.step][key][self.ind]

    def __getitem__(self, key):
        return self.datum(key)

    def __setitem__(self, key, experience):
        self.batches[self.step][key][self.ind] = experience

    def __len__(self):
        return len(self.batches)


def f1(m):
    while True:
        _start = time.time()
        print(m.episode(0)[0]['hi'][0, 0, 0].item(), time.time() - _start, 'get1', len(m))
        time.sleep(3)


def f2(m):
    while True:
        _start = time.time()
        print(m.episode(0)[0]['hi'][0, 0, 0].item(), time.time() - _start, 'get2', len(m))
        time.sleep(3)


if __name__ == '__main__':
    M = Memory()  # Have to compare with numpy shared_memory implementation
    adds = 0
    for _ in range(500):  # Episodes
        for _ in range(1):  # Steps
            d = {'hi': np.random.rand(256, 3, 32, 32), 'done': False}  # Batches
            start = time.time()
            M.add(d)
            adds += time.time() - start
        d = {'hi': np.random.rand(256, 3, 32, 32), 'done': True}  # Last batch
        start = time.time()
        M.add(d)
        adds += time.time() - start
    print(adds, 'adds')
    # M.episode_batches.append(M.episode_batches[-2][-1]['hi'])
    # M.episode_batches[-1].shape = 'la'
    # print(M.episode_batches[-3][-1]['hi'].shape)  # Manager makes copies! Need index list
    l1 = mp.Manager().list([5]*100000)
    l2 = [5]*100000
    start = time.time()
    len(l1)  # len takes too long - if offline maybe cache - shared array of len when add(), can be used for list cache
    print(time.time() - start)
    start = time.time()
    len(l2)
    print(time.time() - start)
    t = torch.as_tensor(100).share_memory_()
    start = time.time()
    b = t[...]
    print(time.time() - start, 'read len')  # A bit faster

    # start = time.time()
    # M.episode_batches[:] = []
    # print(time.time() - start, 'clear manager')  # Slow, maybe only clear after load or so many adds % len

    p1 = mp.Process(name='p1', target=f1, args=(M,))
    p2 = mp.Process(name='p2', target=f2, args=(M,))
    p1.start()
    p2.start()
    d = {'hi': np.random.rand(256, 3, 32, 32), 'done': True}  # Last batch
    start = time.time()
    print(time.time() - start, 'add another')
    start = time.time()
    e = M.episode(0)
    e[0]['hi'] = 5
    print(time.time() - start, 'set')
    e = M.episode(-1)
    e[0]['hi'] = 5
    print(time.time() - start, 'set another')
    # print(e['hi'][:2])
    p1.join()
    p2.join()
