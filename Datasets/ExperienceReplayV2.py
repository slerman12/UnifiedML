# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from math import inf

from torch.utils.data import IterableDataset, Dataset

from Datasets.Memory import Memory


class Replay:
    def __init__(self, path=None, num_workers=1, offline=True, stream=False, batch_size=1, dataset=None, transform=None,
                 gpu_capacity=0, pinned_capacity=0, tensor_ram_capacity=0, ram_capacity=1e6, hd_capacity=inf,
                 gpu_prefetch_factor=0, prefetch_factor=3, pin_memory=False,
                 frame_stack=1, nstep=0, discount=1, meta_shape=(0,)):

        self.memory = Memory(save_path=f'Datasets/ReplayBuffer/{path if path else id(self)}',
                             num_workers=num_workers,
                             gpu_capacity=gpu_capacity,
                             pinned_capacity=pinned_capacity,
                             tensor_ram_capacity=tensor_ram_capacity,
                             ram_capacity=ram_capacity,
                             hd_capacity=hd_capacity)


class Worker:
    def __init__(self, memory):
        self.memory = memory


class Online(Worker, IterableDataset):
    ...


class Offline(Worker, Dataset):
    ...
