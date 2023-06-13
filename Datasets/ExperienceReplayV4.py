# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import os
import random
import threading
from math import inf

import numpy as np

import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp

from Datasets.Memory import Memory
from Datasets.Datasets import load_dataset, to_batch, make_card


class Replay:
    def __init__(self, path=None, num_workers=1, offline=True, stream=False, batch_size=1, dataset=None, transform=None,
                 gpu_capacity=0, pinned_capacity=0, tensor_ram_capacity=0, ram_capacity=1e6, hd_capacity=inf,
                 gpu_prefetch_factor=0, prefetch_factor=3, pin_memory=False,
                 frame_stack=1, nstep=0, discount=1, meta_shape=(0,)):

        # Future steps to compute cumulative reward from
        self.nstep = nstep

        self.epoch = 1

        if stream:
            return

        self.memory = Memory(num_workers=num_workers,
                             gpu_capacity=gpu_capacity,
                             pinned_capacity=pinned_capacity,
                             tensor_ram_capacity=tensor_ram_capacity,
                             ram_capacity=ram_capacity,
                             hd_capacity=hd_capacity)

        # Pytorch Dataset or Memory path
        dataset = load_dataset('Offline', dataset)

        # Fill Memory
        if isinstance(dataset, str):
            # Load Memory from path
            self.memory.load(dataset)
        else:
            # Add Dataset into Memory
            for data in dataset:
                self.memory.add(to_batch(data))

        # Memory save-path TODO
        if '/Offline/' in dataset and not offline:  # Copy to Online
            save_path = ''  # Environment will fill
        else:
            save_path = ''

        self.memory.set_save_path(save_path)
        make_card(save_path)

        # Save to hard disk if Offline
        if isinstance(dataset, Dataset) and offline:
            self.memory.save()

        # DataLoader
        # At exit, maybe save
        # Add for online (including placeholder meta), writable_tape, clear, sample, length, warnings
        # Iterate/next

        # Initialize prefetch tape and transform
        transform = transform  # TODO
        prefetch_tape = PrefetchTape()

        # Parallel worker for batch loading

        worker = ParallelWorker(memory=self.memory,
                                offline=offline,
                                prefetch_tape=prefetch_tape,
                                gpu_prefetch_factor=gpu_prefetch_factor,
                                prefetch_factor=prefetch_factor,
                                pin_memory=pin_memory,
                                transform=transform,
                                frame_stack=frame_stack,
                                nstep=nstep,
                                discount=discount)

        for i in range(num_workers):
            mp.Process(target=worker, args=(i,)).start()

        if num_workers == 0:
            threading.Thread(target=worker).start()


class ParallelWorker:
    def __init__(self, memory, offline, prefetch_tape, gpu_prefetch_factor, prefetch_factor, pin_memory,
                 transform, frame_stack, nstep, discount):
        self.memory = memory

        # Write to shared prefetch tape

    # Worker initialization
    def __call__(self, worker_id):
        seed = np.random.get_state()[1][0] + worker_id
        np.random.seed(seed)
        random.seed(int(seed))
        self.memory.set_worker(worker_id)

    def __len__(self):
        pass


# Index sampler works in multiprocessing
class Sampler:
    def __init__(self, size):
        self.size = size

        self.main_worker = os.getpid()

        self.index = torch.zeros([], dtype=torch.int64).share_memory_()  # Int64

        self.read_lock = mp.Lock()
        self.read_condition = mp.Condition()
        self.index_condition = mp.Condition()

        self.iterator = iter(torch.randperm(self.size))

        threading.Thread(target=self.publish).start()

    # Sample index publisher
    def publish(self):
        assert os.getpid() == self.main_worker, 'Only main worker can feed sample indices.'

        while True:
            with self.read_condition:
                self.read_condition.wait()  # Wait until read is called in a process
                with self.index_condition:
                    self.index[...] = self.next()
                    self.index_condition.notify()  # Notify that index has been updated

    # Sample index subscriber
    def get_index(self):
        with self.read_lock:
            with self.read_condition:
                self.read_condition.notify()  # Notify that read has been called
            with self.index_condition:
                self.index_condition.wait()  # Wait until index has been updated
                return self.index

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(torch.randperm(self.size))
        return next(self.iterator)


class PrefetchTape:  # Can have workers just incrementing this until filled to cap, lowering index w/ each next(replay)
    def __init__(self):
        self.cap = 0

        self.tape = ...

        self.index = torch.zeros([], dtype=torch.int16).share_memory_()  # Int16
        self.lock = mp.Lock()

    def read(self):
        with self.lock:
            index = self.index.item()
            self.index[...] = (index + 1) % self.cap
            return index


mp.set_start_method('spawn')
