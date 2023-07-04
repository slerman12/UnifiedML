# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import atexit
import random
from threading import Thread, Lock
from math import inf
import os

import torchvision
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader

from World.Memory import Memory, Batch
from World.Dataset import load_dataset, datums_as_batch, get_dataset_path, worker_init_fn, compute_stats
from minihydra import instantiate, added_modules, open_yaml, Args


class Replay:
    def __init__(self, path='Replay/', batch_size=1, device='cpu', num_workers=0, offline=True, stream=False,
                 gpu_capacity=0, pinned_capacity=0, ram_capacity=1e6, np_ram_capacity=0, hd_capacity=inf,
                 save=False, mem_size=None, fetch_per=1,
                 prefetch_factor=3, pin_memory=False, pin_device_memory=False, shuffle=True, rewrite_shape=None,
                 dataset=None, transform=None, frame_stack=1, nstep=None, discount=1, agent_specs=None):

        self.device = device
        self.offline = offline
        self.epoch = 1
        self.nstep = nstep or 0  # Future steps to compute cumulative reward from
        self.stream = stream

        if self.stream:
            return

        self.trajectory_flag = Flag()  # Tell worker to include experience trajectories as well

        # CPU workers
        num_workers = max(1, min(num_workers, os.cpu_count()))

        self.memory = Memory(num_workers=num_workers,
                             gpu_capacity=gpu_capacity,
                             pinned_capacity=pinned_capacity,
                             ram_capacity=ram_capacity,
                             np_ram_capacity=np_ram_capacity,
                             hd_capacity=hd_capacity)

        self.rewrite_shape = rewrite_shape  # For rewritable memory

        self.add_lock = Lock()  # For adding to memory in concurrency

        dataset_config = dataset
        card = Args({'_target_': dataset_config}) if isinstance(dataset_config, str) else dataset_config
        # Perhaps if Online, include whether discrete -> continuous, since action shape changes in just that case

        # Optional specs that can be set based on data
        norm, standardize, obs_spec, action_spec = [getattr(agent_specs, spec, None)
                                                    for spec in ['norm', 'standardize', 'obs_spec', 'action_spec']]

        if dataset_config is not None and dataset_config._target_ is not None:
            # TODO Can system-lock w.r.t. save path if load_dataset is Dataset and Offline, then recompute load_dataset
            #   Only one process should save a previously non-existent memory at a time

            if offline:
                root = 'World/ReplayBuffer/Offline/'
                save_path = root + get_dataset_path(dataset_config, root)
            else:
                save_path = 'World/ReplayBuffer/Online/' + path

            # Memory save-path
            self.memory.set_save_path(save_path)

            # Pytorch Dataset or Memory path
            dataset = load_dataset('World/ReplayBuffer/Offline/', dataset_config) if offline else save_path

            # Fill Memory
            if isinstance(dataset, str):
                # Load Memory from path
                if os.path.exists(dataset):
                    self.memory.load(dataset, desc=f'Loading Replay from {dataset}')
                    card = open_yaml(dataset + 'card.yaml')
            else:
                batches = DataLoader(dataset, batch_size=mem_size or batch_size)

                # Add Dataset into Memory in batch-size chunks
                for data in tqdm(batches, desc='Loading Dataset into accelerated Memory...'):
                    self.memory.add(datums_as_batch(data))

            if hasattr(dataset, 'num_classes'):
                card['num_classes'] = dataset.num_classes

            if action_spec is not None and action_spec.discrete:
                if 'discrete_bins' not in action_spec or action_spec.discrete_bins is None:
                    action_spec['discrete_bins'] = card.num_classes

                if 'high' not in action_spec or action_spec.high is None:
                    action_spec['high'] = card.num_classes - 1

                if 'low' not in action_spec or action_spec.low is None:
                    action_spec['low'] = 0
        elif not offline:
            self.memory.set_save_path('World/ReplayBuffer/Online/' + path)

            # Load Memory from path
            dataset = 'World/ReplayBuffer/Online/' + path
            if os.path.exists(dataset):
                self.memory.load(dataset, desc=f'Loading Replay from {dataset}')
                card = open_yaml(dataset + 'card.yaml')

        # Save Online replay on terminate  TODO Maybe delete if not save
        if not offline and save:
            self.memory.set_save_path('World/ReplayBuffer/Online/' + path)
            atexit.register(self.memory.save, desc='Saving Replay Memory...', card=card)

        # TODO Add meta datum if meta_shape, and make sure add() also does - or make dynamic

        added_modules.update({'torchvision': torchvision})
        transform = instantiate(transform)

        # Parallel worker for batch loading

        create_worker = Offline if offline else Online

        worker = create_worker(memory=self.memory,
                               fetch_per=None if offline else fetch_per,
                               transform=transform,
                               frame_stack=frame_stack or 1,
                               nstep=self.nstep,
                               trajectory_flag=self.trajectory_flag,
                               discount=discount)

        # Batch loading

        self.batches = torch.utils.data.DataLoader(dataset=worker,
                                                   batch_size=batch_size,
                                                   num_workers=num_workers,
                                                   pin_memory=pin_memory and 'cuda' in device,  # or pin_device_memory,
                                                   prefetch_factor=prefetch_factor if num_workers else 2,
                                                   shuffle=shuffle and offline,
                                                   worker_init_fn=worker_init_fn,
                                                   persistent_workers=bool(num_workers))

        # Fill in necessary obs_spec and action_spec stats from dataset
        if offline:
            if norm and ('low' not in obs_spec or 'high' not in obs_spec
                         or obs_spec.low is None or obs_spec.high is None) \
                    or standardize and ('mean' not in obs_spec or 'stddev' not in obs_spec
                                        or obs_spec.mean is None or obs_spec.stddev is None):
                if 'stats' not in card:
                    card['stats'] = compute_stats(self.batches)
                if obs_spec is not None:
                    obs_spec.update(card.stats)

        # Save to hard disk if Offline
        if isinstance(dataset, Dataset) and offline:
            # if accelerate and self.memory.num_batches <= sum(self.memory.capacities[:-1]):
            #     root = 'World/ReplayBuffer/Offline/'
            #     self.memory.set_save_path(root + get_dataset_path(dataset_config, root))
            #     save = True
            if save or self.memory.num_batches > sum(self.memory.capacities[:-1]):  # Until save-delete check
                self.memory.save(desc='Memory-mapping Dataset for training acceleration and future re-use. '
                                      'This only has to be done once', card=card)

        # Replay

        self.replay = iter(self)

    # Allows iteration via "next" (e.g. batch = next(replay))
    def __next__(self):
        if self.stream:
            # Environment streaming
            sample = self.stream
        else:
            # Replay sampling
            try:
                sample = next(self.replay)
            except StopIteration:
                self.epoch += 1
                self.replay = iter(self)
                sample = next(self.replay)

        return Batch({key: torch.as_tensor(value, device=self.device).to(non_blocking=True)
                      for key, value in sample.items()})

    def __iter__(self):
        self.replay = iter(self.batches)
        return self.replay

    def include_trajectories(self):
        self.trajectory_flag.set()

    def add(self, trace):
        if trace is None:
            trace = []

        for batch in trace:
            if self.stream:
                self.stream = batch  # For streaming directly from Environment  TODO N-step in {0, 1}
            else:
                def add():
                    with self.add_lock:
                        self.memory.add(batch)  # Add to memory

                Thread(target=add).start()  # Threading

    def set_tape(self, shape):
        self.rewrite_shape = shape or [0]

    def writable_tape(self, batch, ind, step):
        assert isinstance(batch, (dict, Batch)), f'expected \'batch\' to be dict or Batch, got {type(batch)}.'
        self.memory.writable_tape(batch, ind, step)

    def __len__(self):
        # Infinite if stream, else num episodes in Memory
        return int(9e9) if self.stream else len(self.memory)


class Worker:
    def __init__(self, memory, fetch_per, transform, frame_stack, nstep, trajectory_flag, discount):
        self.memory = memory
        self.fetch_per = fetch_per

        self.samples_since_last_fetch = 0

        self.transform = transform

        self.frame_stack = frame_stack
        self.nstep = nstep
        self.trajectory_flag = trajectory_flag
        self.discount = discount

        self.initialized = False

    @property
    def worker(self):
        try:
            return torch.utils.data.get_worker_info().id
        except AttributeError:
            return 0

    def sample(self, index=None, update=False):
        if not self.initialized:
            self.memory.set_worker(self.worker)
            self.initialized = True

        # Periodically update memory
        while self.fetch_per and not self.samples_since_last_fetch % self.fetch_per or update:
            self.memory.update()

            if len(self.memory):
                break

        self.samples_since_last_fetch += 1

        _index = index

        # Sample index
        if index is None:
            index = random.randint(0, len(self.memory) - 1)  # Random sample an episode

        # Retrieve from Memory
        episode = self.memory[index]

        # nstep = bool(self.nstep)  # Allows dynamic nstep
        nstep = self.nstep  # But w/o step as input, models can't distinguish later episode steps

        if len(episode) < nstep + 1:  # Make sure at least one nstep is present if nstep
            return self.sample(_index, update=True)

        step = random.randint(0, len(episode) - 1 - nstep)  # Randomly sample experience in episode
        experience = Args(episode[step])

        # Frame stack / N-step
        experience = self.compute_RL(episode, experience, step)

        # Transform
        if self.transform is not None:
            experience.obs = self.transform(experience.obs)

        # Add metadata
        experience['episode_index'] = index
        experience['episode_step'] = step

        if 'label' in experience and experience.label.dtype == torch.int64:
            experience.label = experience.label.long()

        return experience

    def compute_RL(self, episode, experience, step):
        # TODO Just apply nstep and frame stack as transforms nstep, frame_stack, transform

        # Frame stack
        def frame_stack(traj, key, idx):
            frames = traj[max([0, idx + 1 - self.frame_stack]):idx + 1]
            for _ in range(self.frame_stack - idx - 1):  # If not enough frames, reuse the first
                frames = traj[:1] + frames
            frames = torch.concat([torch.as_tensor(frame[key])
                                   for frame in frames]).reshape(frames[0][key].shape[0] * self.frame_stack,
                                                                 *frames[0][key].shape[1:])
            return frames

        # Present
        if self.frame_stack > 1:
            experience.obs = frame_stack(episode, 'obs', step)  # Need experience as own dict/Batch for this

        # Future
        if self.nstep:
            # Transition
            experience.action = episode[step + 1].action

            traj_r = torch.as_tensor([experience.reward
                                      for experience in episode[step + 1:step + self.nstep + 1]])

            experience['next_obs'] = frame_stack(episode, 'obs', step + len(traj_r))

            # Trajectory TODO
            if self.trajectory_flag:
                experience['traj_r'] = traj_r
                traj_o = np.concatenate([episode['obs'][max(0, idx - i):max(idx + self.nstep + 1 - i, self.nstep + 1)]
                                         for i in range(self.frame_stack - 1, -1, -1)], 1)  # Frame_stack
                traj_a = episode['action'][idx + 1:idx + self.nstep + 1]
                if 'label' in experience:
                    traj_l = episode['label'][idx:idx + self.nstep + 1]

            # Cumulative discounted reward
            discounts = self.discount ** np.arange(len(traj_r) + 1)
            experience.reward = np.dot(discounts[:-1], traj_r).astype('float32')
            experience['discount'] = 0 if episode[step + len(traj_r)].done else discounts[-1].astype('float32')
        else:
            experience['discount'] = 1

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
            self._flag = bool(self.flag)
        return self._flag
