# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import os
import random
import threading
from functools import cached_property
from math import inf

import numpy as np

import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp

from Datasets.Memory import Memory, Batch
from Datasets.Datasets import load_dataset, to_experience, make_card


class Replay:
    def __init__(self, path=None, batch_size=1, device='cpu', num_workers=1, offline=True, stream=False,
                 gpu_capacity=0, pinned_capacity=0, tensor_ram_capacity=0, ram_capacity=1e6, hd_capacity=inf,
                 gpu_prefetch_factor=0, prefetch_factor=3, pin_memory=False,
                 dataset=None, transform=None, frame_stack=1, nstep=0, discount=1, meta_shape=(0,)):

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
                self.memory.add(to_experience(data))

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

        # Initialize sampler, prefetch tape, and transform
        sampler = Sampler(size=len(self.memory)) if offline else None
        prefetch_tape = PrefetchTape(batch_size=batch_size,
                                     device=device,
                                     gpu_prefetch_factor=gpu_prefetch_factor,
                                     prefetch_factor=prefetch_factor,
                                     pin_memory=pin_memory)
        transform = transform  # TODO

        # Parallel worker for batch loading

        worker = ParallelWorker(memory=self.memory,
                                offline=offline,
                                sampler=sampler,
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
            threading.Thread(target=worker, args=(0,)).start()


class ParallelWorker:
    def __init__(self, memory, offline,sampler,  prefetch_tape, gpu_prefetch_factor, prefetch_factor, pin_memory,
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


class PrefetchTape:
    def __init__(self, batch_size, device, gpu_prefetch_factor=0, prefetch_factor=3, pin_memory=False):
        assert gpu_prefetch_factor + prefetch_factor > 0

        self.batch_size = batch_size
        self.device = device

        self.cap = batch_size * (gpu_prefetch_factor + prefetch_factor)
        self.gpu_prefetch_factor = gpu_prefetch_factor
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

        self.batches = [torch.zeros([], dtype=torch.int16)]

        self.start_index = torch.zeros([], dtype=torch.int16).share_memory_()  # Int16
        self.end_index = torch.ones([], dtype=torch.int16).share_memory_()  # Int16
        self.index_lock = mp.Lock()

        self.cuda_prefetch_tape = mp.Manager().dict()
        self.ram_prefetch_tape = mp.Manager().dict()

        self.init_lock = mp.Lock()

        self.initialized = torch.tensor(False).share_memory_()

    def create(self, experience):
        if not self.initialized:
            with self.init_lock:
                if not self.initialized:
                    if self.gpu_prefetch_factor:
                        for key, datum in experience.items():
                            self.cuda_prefetch_tape[key] = list(torch.zeros(self.batch_size * self.gpu_prefetch_factor,
                                                                            *datum.shape).cuda(non_blocking=True
                                                                                               ).unbind())

                    if self.prefetch_factor:
                        ram_prefetch_tape = {key: list(torch.zeros(self.batch_size * self.prefetch_factor,
                                                                   *datum.shape).unbind())
                                             for key, datum in experience.items()}

                        for key, datum in ram_prefetch_tape.items():
                            if self.pin_memory:
                                self.ram_prefetch_tape[key] = [datum.to(non_blocking=True).pin_memory()
                                                               for datum in ram_prefetch_tape[key]]
                            else:
                                self.ram_prefetch_tape[key] = ram_prefetch_tape[key]

                    self.initialized[...] = True
                    self.initialized = True

    @cached_property
    def prefetch_tape(self):
        return dict(self.cuda_prefetch_tape), dict(self.ram_prefetch_tape)

    def check_if_full(self):
        if self.end_index > self.start_index:
            return self.end_index - self.start_index + self.batch_size > self.cap
        else:
            return self.cap - self.start_index + self.end_index + self.batch_size > self.cap

    def add(self, experience):  # Currently assumes all experiences consist of the same keys
        cuda_prefetch_tape, ram_prefetch_tape = self.prefetch_tape
        index = self.index.item()

        if index >

        self.prefetch_tape[self.index] = experience

    @property
    def index(self):
        with self.index_lock:
            index = self.end_index.item() - 1
            self.end_index[...] = (index + 1) % self.cap
            return index

    def get_batch(self):
        # Get tape if needed

        end_index = (self.start_index + self.batch_size) % self.end_index

        if self.end_index > self.start_index:
            experiences = self.prefetch_tape[self.start_index:end_index]
        else:
            experiences = self.prefetch_tape[self.start_index:] + self.prefetch_tape[:end_index]

        self.start_index[...] = end_index

        # Collate
        batch = {key: torch.stack([torch.as_tensor(datum).to(self.device, non_blocking=True)
                                   for datum in [experience[key] for experience in experiences]])
                 for key in experiences[0]}

        return Batch(batch)


mp.set_start_method('spawn')
