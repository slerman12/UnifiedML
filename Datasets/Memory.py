# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import numpy as np

import torch
import torch.multiprocessing as mp


debug_toggle = False  # Can /probably/ use


class Memory:
    def __init__(self, device=None, device_capacity=None, ram_capacity=None, cache_capacity=None, hd_capacity=None):
        self.path = './ReplayBuffer'

        manager = mp.Manager()

        self.index = manager.list()
        self.episode_batches = manager.list()

    def add(self, batch):
        if batch['done'] or not self.episode_batches:
            self.episode_batches.append(())

        for key in batch:
            batch[key] = torch.as_tensor(batch[key]).share_memory_()  # .to(non_blocking=True)

        self.episode_batches[-1] = self.episode_batches[-1] + (batch,)

        batch_size = 1

        for mem in batch.values():
            if mem.shape and len(mem) > 1:
                batch_size = len(mem)
                break

        episode_batches_ind = len(self.episode_batches) - 1

        self.index.extend(enumerate([episode_batches_ind] * batch_size))

    def episode(self, ind):
        if debug_toggle:
            return Episode(self, ind)
        else:
            ind, episode_batches_ind = self.index[ind]
            batches = self.episode_batches[episode_batches_ind]

            # Return experiences in episode
            return [{key: value[ind] if value.shape else value for key, value in batch.items()} for batch in batches]
            # return [self.experience(ind, step) for step, _ in enumerate(batches)]

    def experience(self, ind, step):
        if not debug_toggle:
            ind, episode_batches_ind = self.index[ind]
            batches = self.episode_batches[episode_batches_ind]

            # Return experience in episode
            return {key: value[ind] if value.shape else value for key, value in batches[step].items()}

    def __getitem__(self, ind):
        # return self.episode(ind)

        if debug_toggle:
            return Episode(self, ind)
        else:
            ind, episode_batches_ind = self.index[ind]
            batches = self.episode_batches[episode_batches_ind]

            # Return experiences in episode - For some reason this way is the fastest
            return [{key: value[ind] if value.shape else value for key, value in batch.items()} for batch in batches]

    def __len__(self):
        return len(self.index)


class Episode:
    def __init__(self, memory, ind):
        self.memory = memory
        self.ind = ind

    @property
    def batches(self):
        ind, episode_batches_ind = self.memory.index[self.ind]
        return self.memory.episode_batches[episode_batches_ind]

    def datum(self, key):
        return Experience(self, key)

    def __getitem__(self, key):
        return self.datum(key)

    def __setitem__(self, key, value):
        for batch in self.batches:
            batch[key][self.ind] = value  # self.ind refers to wrong ind

    def __len__(self):
        return len(self.batches)


class Experience:
    def __init__(self, episode, key):
        self.episode = episode
        self.key = key

    def time_step(self, time_step):
        return self.episode.batches[time_step][self.key][self.episode.ind]

    def __getitem__(self, time_step):
        return self.time_step(time_step)

    def __setitem__(self, time_step, experience):
        self.episode.batches[time_step][self.key][self.episode.ind] = experience

    def len(self):
        return len(self.episode)


class MMAP:
    def __init__(self, datum, path=None):
        if isinstance(datum, np.memmap):
            self.datum = datum
        else:
            datum = torch.as_tensor(datum).numpy()

    def __getitem__(self, item):
        return MMAP(item)

    def __setitem__(self, key, value):
        pass

    def as_tensor(self):
        pass


def f1(m):
    while True:
        _start = time.time()
        if debug_toggle:
            print(m.episode(0)['hi'][0][0, 0, 0].item(), time.time() - _start, 'get1', len(m))
        else:
            print(m[0][0]['hi'][0, 0, 0].item(), time.time() - _start, 'get1', len(m))
            # print(m.episode(0)[0]['hi'][0, 0, 0].item(), time.time() - _start, 'get1', len(m))
            # print(m.experience(0, 0)['hi'][0, 0, 0].item(), time.time() - _start, 'get1', len(m))
        time.sleep(3)


def f2(m):
    while True:
        _start = time.time()
        if debug_toggle:
            print(m.episode(0)['hi'][0][0, 0, 0].item(), time.time() - _start, 'get2', len(m))
        else:
            print(m[0][0]['hi'][0, 0, 0].item(), time.time() - _start, 'get1', len(m))
            # print(m.episode(0)[0]['hi'][0, 0, 0].item(), time.time() - _start, 'get1', len(m))
            # print(m.experience(0, 0)['hi'][0, 0, 0].item(), time.time() - _start, 'get2', len(m))
        time.sleep(3)


if __name__ == '__main__':
    M = Memory()
    adds = 0
    for _ in range(5):  # Episodes
        for _ in range(128 - 1):  # Steps
            d = {'hi': np.ones([256, 3, 32, 32]), 'done': False}  # Batches
            start = time.time()
            M.add(d)
            adds += time.time() - start
        done = {'hi': np.ones([256, 3, 32, 32]), 'done': True}  # Last batch
        start = time.time()
        M.add(done)
        adds += time.time() - start
    print(adds, 'adds')
    p1 = mp.Process(name='p1', target=f1, args=(M,))
    p2 = mp.Process(name='p', target=f2, args=(M,))
    p1.start()
    p2.start()
    start = time.time()
    if debug_toggle:
        e = M.episode(0)
        e['hi'][0][0] = 5
    else:
        M[0][0]['hi'][0] = 5
        # M.episode(0)[0]['hi'][0] = 5
        # M.experience(0, 0)['hi'][0] = 5
    print(time.time() - start, 'set')
    p1.join()
    p2.join()
