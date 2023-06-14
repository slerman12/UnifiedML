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

from Datasets.Memory import Memory, Batch
from Datasets.Datasets import load_dataset, to_experience, make_card
from Hyperparams.minihydra import instantiate


class Replay:
    def __init__(self, path=None, batch_size=1, device='cpu', num_workers=1, offline=True, stream=False,
                 gpu_capacity=0, pinned_capacity=0, tensor_ram_capacity=0, ram_capacity=1e6, hd_capacity=inf,
                 gpu_prefetch_factor=0, prefetch_factor=3, pin_memory=False,
                 dataset=None, transform=None, frame_stack=1, nstep=0, discount=1, meta_shape=(0,)):

        # Future steps to compute cumulative reward from
        self.nstep = nstep

        self.stream = stream

        if self.stream:
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

        # TODO At exit, maybe save

        # TODO Add meta datum if meta_shape, and make sure add() also does - or make dynamic

        # Initialize sampler, prefetch tape, and transform
        self.sampler = Sampler(size=len(self.memory)) if offline else None
        self.prefetch_tape = PrefetchTape(batch_size=batch_size,
                                          device=device,
                                          gpu_prefetch_factor=gpu_prefetch_factor,
                                          prefetch_factor=prefetch_factor,
                                          pin_memory=pin_memory)
        transform = instantiate(transform)

        # Parallel worker for batch loading

        worker = ParallelWorker(memory=self.memory,
                                sampler=self.sampler,
                                prefetch_tape=self.prefetch_tape,
                                transform=transform,
                                frame_stack=frame_stack,
                                nstep=nstep,
                                discount=discount)

        for i in range(num_workers):
            mp.Process(target=worker, args=(i,)).start()

        if num_workers == 0:
            threading.Thread(target=worker, args=(0,)).start()

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

    @property
    def epoch(self):
        return self.sampler.epoch

    def sample(self, trajectories=False):
        if self.stream:
            # Streaming
            return [self.stream.get(key, torch.empty([1, 0]))
                    for key in ['obs', 'action', 'reward', 'discount', 'next_obs', 'label', *[None] * 4 * trajectories,
                                'step', 'ids', 'meta']]  # Return contents of the data stream
        else:
            # Get batch from prefetch tape
            batch = self.prefetch_tape.get_batch()
            return *batch[:10 if trajectories else 6], *batch[10:]  # Return batch, w(/o) future-trajectories

    def add(self, batch):
        if self.stream:
            self.stream = batch  # For streaming directly from Environment  TODO N-step in {0, 1}
        else:
            self.memory.add(batch)

    def writable_tape(self, batch, ind, step):
        assert isinstance(batch, (dict, Batch)), f'expected \'batch\' to be dict or Batch, got {type(batch)}.'
        self.memory.writable_tape(batch, ind, step)

    def __len__(self):
        # Infinite if stream, else num episodes in Memory
        return int(5e11) if self.stream else len(self.memory)


class ParallelWorker:
    def __init__(self, memory, sampler,  prefetch_tape, transform, frame_stack, nstep, discount):
        self.memory = memory
        self.sampler = sampler
        self.prefetch_tape = prefetch_tape

        self.transform = transform

        self.frame_stack = frame_stack
        self.nstep = nstep
        self.discount = discount

    # Worker initialization
    def __call__(self, worker):
        seed = np.random.get_state()[1][0] + worker
        np.random.seed(seed)
        random.seed(int(seed))
        self.memory.set_worker(worker)

        while True:
            # Get index from sampler
            if self.sampler is None:
                index = random.randint(0, len(self.memory))  # Random sample an episode
            else:
                index = self.sampler.get_index()

            # Retrieve experience from Memory
            episode = self.memory[index]
            index = random.randint(0, len(episode) - self.nstep)  # Randomly sample sub-episode
            experience = episode[index]

            # Transform / N-step / frame stack
            experience = self.compute_RL(episode, experience, index)
            experience.obs = self.transform(torch.as_tensor(experience.obs))

            # Add to prefetch tape (halts if full)
            self.prefetch_tape.add(experience)

    def compute_RL(self, episode, experience, index):
        return experience


# Index sampler works in multiprocessing
class Sampler:
    def __init__(self, size):
        self.size = size

        self.epoch = torch.zeros([], dtype=torch.int32).share_memory_()  # Int32

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
            self.epoch[...] = self.epoch + 1
        return next(self.iterator)


class PrefetchTape:
    def __init__(self, batch_size, device, gpu_prefetch_factor=0, prefetch_factor=3, pin_memory=False):
        self.batch_size = batch_size
        self.cap = batch_size * (gpu_prefetch_factor + prefetch_factor)
        self.device = device

        self.prefetch_tape = Memory(gpu_capacity=batch_size * gpu_prefetch_factor,
                                    pinned_capacity=batch_size * prefetch_factor * pin_memory,
                                    tensor_ram_capacity=batch_size * prefetch_factor * (not pin_memory),
                                    ram_capacity=0,
                                    hd_capacity=0)

        self.batches = [torch.zeros([], dtype=torch.int16)]

        self.start_index = torch.zeros([], dtype=torch.int16).share_memory_()  # Int16
        self.end_index = torch.ones([], dtype=torch.int16).share_memory_()  # Int16
        self.add_lock = mp.Lock()
        self.index_lock = mp.Lock()

    def add(self, experience):  # Currently assumes all experiences consist of the same keys
        if not len(self.prefetch_tape):
            with self.add_lock:
                if not len(self.prefetch_tape):
                    for _ in range(self.cap):
                        self.prefetch_tape.add(experience)

        device = None
        index = self.index()

        for datum in self.prefetch_tape[index].values():
            if hasattr(datum, 'device'):
                device = datum.device
                break

        self.prefetch_tape[index] = {key: torch.as_tensor(datum)[None, :].to(device, non_blocking=True)
                                     for key, datum in experience.items()}

    def full(self, index):
        if index > self.start_index:
            return index - self.start_index + self.batch_size > self.cap
        else:
            return self.cap - self.start_index + index + self.batch_size > self.cap

    def index(self):
        with self.index_lock:
            index = self.end_index.item() - 1

            while self.full(index + 1):
                pass

            self.end_index[...] = (index + 1) % self.cap
            return index

    def get_batch(self):
        end_index = (self.start_index + self.batch_size) % self.end_index

        if end_index > self.start_index:
            experiences = self.prefetch_tape[self.start_index:end_index]
        else:
            experiences = self.prefetch_tape[self.start_index:] + self.prefetch_tape[:end_index]

        self.start_index[...] = end_index

        # Collate
        batch = {key: torch.concat([torch.as_tensor(datum).to(self.device, non_blocking=True)
                                    for datum in [experience[key] for experience in experiences]])
                 for key in experiences[0]}

        return Batch(batch)


mp.set_start_method('spawn')
