# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import atexit
import random
from math import inf

from torch.utils.data.dataset import T_co
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from World.Memory import Memory, Batch
from World.Dataset import load_dataset, datums_as_batch, get_dataset_path, Transform
from Hyperparams.minihydra import instantiate, Args


class Replay:
    def __init__(self, path='Replay/', save=True, batch_size=1, device='cpu', num_workers=0, offline=True, stream=False,
                 gpu_capacity=0, pinned_capacity=0, tensor_ram_capacity=0, ram_capacity=1e6, hd_capacity=inf,
                 mem_size=None, prefetch_factor=3, pin_memory=False, pin_device_memory=False, reload=True, shuffle=True,
                 dataset=None, transform=None, frame_stack=1, nstep=None, discount=1, meta_shape=(0,), **kwargs):

        self.epoch = 1
        self.nstep = nstep or 0  # Future steps to compute cumulative reward from
        self.stream = stream

        if self.stream:
            return

        self.reload = reload

        self.memory = Memory(num_workers=num_workers,
                             gpu_capacity=gpu_capacity,
                             pinned_capacity=pinned_capacity,
                             tensor_ram_capacity=tensor_ram_capacity,
                             ram_capacity=ram_capacity,
                             hd_capacity=hd_capacity)

        dataset_config = dataset
        card = {'_target_': dataset_config} if isinstance(dataset_config, str) else dataset_config

        # TODO System-wide lock perhaps, w.r.t. Offline Dataset save path if load_dataset is Dataset and Offline
        if dataset_config is not None:
            # Pytorch Dataset or Memory path
            dataset = load_dataset('World/ReplayBuffer/Offline/', dataset_config, **kwargs)

            # TODO Can system-lock w.r.t. save path if load_dataset is Dataset and Offline, then recompute load_dataset

            if offline:
                root = 'World/ReplayBuffer/Offline/'
                save_path = root + get_dataset_path(dataset_config, root)
            else:
                save_path = 'World/ReplayBuffer/Online/' + path

            # Memory save-path
            self.memory.set_save_path(save_path)

            # Fill Memory
            if isinstance(dataset, str):
                # Load Memory from path
                self.memory.load(dataset, desc=f'Loading Replay from {dataset}')

                if not offline and dataset != 'World/ReplayBuffer/Online/' + path:
                    self.memory.saved(False, desc='Setting saved flag of Online version of Offline Replay to False')
            else:
                batches = DataLoader(Transform(dataset), batch_size=mem_size or batch_size)

                # Add Dataset into Memory in batch-size chunks
                for data in tqdm(batches, desc='Loading Dataset into accelerated Memory...'):
                    self.memory.add(datums_as_batch(data))

            if save_path:
                # Save to hard disk if Offline
                if isinstance(dataset, Dataset) and offline:
                    # if accelerate and self.memory.num_batches <= sum(self.memory.capacities[:-1]):
                    #     root = 'World/ReplayBuffer/Offline/'
                    #     self.memory.set_save_path(root + get_dataset_path(dataset_config, root))
                    #     save = True
                    if save or self.memory.num_batches > sum(self.memory.capacities[:-1]):  # Until save-delete check
                        self.memory.save(desc='Memory-mapping Dataset for training acceleration and future re-use. '
                                              'This only has to be done once', card=card)

        # Save Online replay on terminate  Maybe delete if not save
        if not offline and save:
            atexit.register(lambda: self.memory.save(desc='Saving Replay Memory...', card=card))

        # TODO Add meta datum if meta_shape, and make sure add() also does - or make dynamic

        transform = instantiate(transform)

        # Parallel worker for batch loading

        create_worker = Offline if offline else Online

        worker = create_worker(memory=self.memory,
                               transform=transform,
                               frame_stack=frame_stack,
                               nstep=self.nstep,
                               discount=discount)

        # Batch loading

        self.batches = torch.utils.data.DataLoader(dataset=worker,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   pin_memory=pin_memory and 'cuda' in device and not pin_device_memory,
                                                   prefetch_factor=prefetch_factor if num_workers else 2,
                                                   shuffle=shuffle and offline,
                                                   worker_init_fn=worker_init_fn,
                                                   persistent_workers=bool(num_workers))

        # Replay

        self._replay = None

    # Allows iteration via "next" (e.g. batch = next(replay))
    def __next__(self):
        return self.sample()

    # Samples stored experiences or pulls from a data stream, optionally includes trajectories, returns a batch
    def sample(self, trajectories=False):
        if self.stream:
            # Streaming
            return [self.stream.get(key, torch.empty([1, 0]))
                    for key in ['obs', 'action', 'reward', 'discount', 'next_obs', 'label', *[None] * 4 * trajectories,
                                'step', 'ids', 'meta']]  # Return contents of the data stream
        else:
            # Sampling
            try:
                sample = next(self.replay)
            except StopIteration as stop:
                if not self.reload:
                    raise stop
                self.epoch += 1
                self._replay = None  # Reset iterator when depleted
                sample = next(self.replay)
            return sample
            # return *sample[:10 if trajectories else 6], *sample[10:]  # Return batch, w(/o) future-trajectories

    # Initial iterator, allows replay iteration
    def __iter__(self):
        self._replay = iter(self.batches)
        return self.replay

    @property
    def replay(self):
        if self._replay is None:
            self._replay = iter(self.batches)  # Recreates the iterator when exhausted
        return self._replay

    def add(self, batch):
        if self.stream:
            self.stream = batch  # For streaming directly from Environment  TODO N-step in {0, 1}
        else:
            self.memory.add(batch)

    def writable_tape(self, batch, ind, step):
        assert isinstance(batch, (dict, Batch)), f'expected \'batch\' to be dict or Batch, got {type(batch)}.'
        self.memory.writable_tape(batch, ind, step)

    def __len__(self):
        if not self.reload:
            return len(self.batches)

        # Infinite if stream, else num episodes in Memory
        return int(5e11) if self.stream else len(self.memory)


def worker_init_fn(worker_id):
        seed = np.random.get_state()[1][0] + worker_id
        np.random.seed(seed)
        random.seed(int(seed))


class Worker:
    def __init__(self, memory, transform, frame_stack, nstep, discount):
        self.memory = memory

        self.transform = transform

        self.frame_stack = frame_stack
        self.nstep = nstep
        self.discount = discount

        self.initialized = False

    @property
    def worker(self):
        try:
            return torch.utils.data.get_worker_info().id
        except AttributeError:
            return 0

    def sample(self, index=None):
        if not self.initialized:
            print(len(self.memory), self.worker)
            self.memory.set_worker(self.worker)
            self.initialized = True

        # Sample index
        if index is None:
            index = random.randint(0, len(self.memory))  # Random sample an episode

        # Retrieve from Memory
        episode = self.memory[index]
        step = random.randint(0, len(episode) - self.nstep - 1)  # Randomly sample sub-episode
        experience = Args(episode[step])

        # Frame stack / N-step
        experience = self.compute_RL(episode, experience, step)

        # Transform
        if self.transform is not None:
            experience.obs = self.transform(experience.obs)

        # Add metadata
        experience['episode_index'] = index
        experience['episode_step'] = step

        return experience

    def compute_RL(self, episode, experience, step):

        # Frame stack
        def frame_stack(traj, key, idx):
            frames = traj[max([0, idx + 1 - self.frame_stack]):idx + 1]
            for _ in range(self.frame_stack - idx - 1):  # If not enough frames, reuse the first
                frames = traj[:1] + frames
            frames = torch.concat([torch.as_tensor(frame[key])
                                   for frame in frames]).reshape(frames.shape[1] * self.frame_stack, *frames.shape[2:])
            return frames

        # Present
        if self.frame_stack > 1:
            experience.obs = frame_stack(experience, 'obs', step)  # Need experience as own dict/Batch for this

        # Future
        if self.nstep:
            # Transition
            experience.action = episode[step + 1].action
            experience['next_obs'] = frame_stack(episode, 'obs', step + self.nstep)

            # Trajectory TODO
            # traj_o = np.concatenate([episode['obs'][max(0, idx - i):max(idx + self.nstep + 1 - i, self.nstep + 1)]
            #                          for i in range(self.frame_stack - 1, -1, -1)], 1)  # Frame_stack
            # traj_a = episode['action'][idx + 1:idx + self.nstep + 1]
            traj_r = torch.as_tensor([experience.reward
                                      for experience in episode[step + 1:step + self.nstep + 1]])
            # traj_l = episode['label'][idx:idx + self.nstep + 1]

            # Cumulative discounted reward
            discounts = self.discount ** np.arange(self.nstep + 1)
            experience.reward = np.dot(discounts[:-1], traj_r)
            experience['discount'] = discounts[-1:]
        else:
            experience['discount'] = 1.0

        return experience

    def __len__(self):
        return len(self.memory)


class Offline(Worker, Dataset):
    def __getitem__(self, index):
        return self.sample(index)  # Retrieve a single experience by index


class Online(Worker, IterableDataset):
    def __iter__(self):
        while True:
            yield self.sample()  # Yields a single experience


# Quick parallel one-time flag
class Flag:
    def __init__(self):
        self.flag = torch.tensor(False, dtype=torch.bool).share_memory_()
        self._flag = False

    def set(self):
        self.flag[...] = self._flag = True

    def __bool__(self):
        if not self._flag:
            self._flag = self.flag
        return self._flag
