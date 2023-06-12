# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import random
from math import inf

import numpy as np

from torch.utils.data import IterableDataset, Dataset, DataLoader

from Hyperparams.minihydra import Args, valid_path

from Datasets.Memory import Memory


class ExperienceReplay:
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

        data = dataset

        # Extract data by config
        if isinstance(data, Args):
            if valid_path(data._target_):
                data = data._target_
            else:
                assert valid_path(data._target_, instantiation_path=True)
                data = load_dataset(data)  # data = Pytorch Dataset or string path

        # Extract data by string
        if isinstance(data, str):
            if valid_path(data, instantiation_path=True):
                data = load_dataset(data)  # data = Pytorch Dataset or string path
            else:
                assert valid_path(data)

        # Load data into memory
        if isinstance(data, Dataset):
            # Add into memory
            pass
        else:
            # Load into memory
            self.memory.load(data)

        # Separate Offline and Online, and save to hard disk if Offline
        if isinstance(data, Dataset):
            if offline:
                save_path = ''
                # Save .yaml in save_path
                self.memory.set_save_path(save_path)
                self.memory.save()
            else:
                save_path = ''
                # Save .yaml in save_path
                self.memory.set_save_path(save_path)
        else:
            if '/Offline/' in data and not offline:
                save_path = ''
                # Copy .yaml in save_path
                self.memory.set_save_path(save_path)  # Copy to Online

        # DataLoader
        # At exit, maybe save
        # Add for online (including placeholder meta), writable_tape, clear, sample, length, warnings
        # Iterate/next

        # Initialize prefetch tape and transform
        transform = transform  # TODO
        prefetch_tape = PrefetchTape()

        # Parallel worker for batch loading

        worker = (Offline if offline else Online)(memory=self.memory,
                                                  offline=offline,
                                                  prefetch_tape=prefetch_tape,
                                                  gpu_prefetch_factor=gpu_prefetch_factor,
                                                  prefetch_factor=prefetch_factor,
                                                  pin_memory=pin_memory,
                                                  transform=transform,
                                                  frame_stack=frame_stack,
                                                  nstep=nstep,
                                                  discount=discount)

        # Batch loading

        self.batches = DataLoader(dataset=worker,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=offline,
                                  worker_init_fn=InitializeWorker(self.memory),
                                  persistent_workers=True)


class ParallelWorker:
    def __init__(self, memory, offline, prefetch_tape, gpu_prefetch_factor, prefetch_factor, pin_memory,
                 transform, frame_stack, nstep, discount):
        self.memory = memory

        # Write to shared prefetch tape


class Online(ParallelWorker, IterableDataset):
    ...


class Offline(ParallelWorker, Dataset):
    ...


class InitializeWorker:
    def __init__(self, memory):
        self.memory = memory

    def __call__(self, worker_id):
        seed = np.random.get_state()[1][0] + worker_id
        np.random.seed(seed)
        random.seed(int(seed))
        self.memory.set_worker(worker_id)


# Custom sampler for prefetch-tape
