# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import atexit
import os
import random
import threading
from math import inf

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

from World.Memory import Memory, Batch
from World.Dataset import load_dataset, datums_as_batch, get_dataset_path, Transform, add_batch_dim
from Hyperparams.minihydra import instantiate, Args


class Replay:
    def __init__(self, path='Replay/', save=True, batch_size=1, device='cpu', num_workers=1, offline=True, stream=False,
                 gpu_capacity=0, pinned_capacity=0, tensor_ram_capacity=1e6, ram_capacity=0, hd_capacity=inf,
                 gpu_prefetch_factor=0, prefetch_factor=3, pin_memory=False, mem_size=None,
                 dataset=None, transform=None, frame_stack=1, nstep=0, discount=1, meta_shape=(0,), **kwargs):

        # Future steps to compute cumulative reward from
        self.nstep = nstep or 0

        self.stream = stream

        if self.stream:
            return

        self.memory = Memory(num_workers=num_workers,
                             gpu_capacity=gpu_capacity,
                             pinned_capacity=pinned_capacity,
                             tensor_ram_capacity=tensor_ram_capacity,
                             ram_capacity=ram_capacity,
                             hd_capacity=hd_capacity)

        dataset_config = dataset
        card = {'_target_': dataset_config} if isinstance(dataset_config, str) else dataset_config

        # TODO System-wide lock perhaps, w.r.t. Offline Dataset save path if load_dataset is Dataset and Offline
        if dataset is not None:
            # Pytorch Dataset or Memory path
            dataset = load_dataset('World/ReplayBuffer/Offline/', dataset_config, **kwargs)

            # TODO Can system-lock w.r.t. save path if load_dataset is Dataset and Offline, then recompute load_dataset

            # Memory save-path
            save_path = None

            if isinstance(dataset, Dataset) and offline:
                root = 'World/ReplayBuffer/Offline/'
                save_path = root + get_dataset_path(dataset_config, root)
            elif not offline:
                save_path = 'World/ReplayBuffer/Online/' + path

            if save_path:
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
                    self.memory.save(desc='Memory-mapping Dataset for training acceleration and future re-use. '
                                          'This only has to be done once', card=card)

        # Save Online replay on terminate  Maybe delete if not save
        if not offline and save:
            atexit.register(lambda: self.memory.save(desc='Saving Replay Memory...', card=card))

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
        return self.sampler.epoch.item()

    def sample(self, trajectories=False):
        if self.stream:
            # Streaming
            return [self.stream.get(key, torch.empty([1, 0]))
                    for key in ['obs', 'action', 'reward', 'discount', 'next_obs', 'label', *[None] * 4 * trajectories,
                                'step', 'ids', 'meta']]  # Return contents of the data stream
        else:
            # Get batch from prefetch tape
            batch = self.prefetch_tape.get_batch()
            return batch
            # return *batch[:10 if trajectories else 6], *batch[10:]  # Return batch, w(/o) future-trajectories

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

        while True:  # TODO workers should periodically update if online
            # Sample index
            if self.sampler is None:
                index = random.randint(0, len(self.memory))  # Random sample an episode
            else:
                index = self.sampler.get_index()

            # Retrieve from Memory
            episode = self.memory[index]
            step = random.randint(0, len(episode) - self.nstep - 1)  # Randomly sample sub-episode
            experience = Args(episode[step])

            # Frame stack / N-step
            experience = self.compute_RL(episode, experience, step)

            # Transform
            if self.transform is not None:
                experience.obs = self.transform(torch.as_tensor(experience.obs))

            # Add metadata
            experience['episode_index'] = index
            experience['episode_step'] = step

            # Add to prefetch tape (halts if full)
            self.prefetch_tape.add(experience)

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

    # Sample index publisher TODO Event to crash when main crashes!
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
                    self.prefetch_tape.main_worker = os.getpid()
                    for _ in range(self.cap):
                        self.prefetch_tape.add({key: add_batch_dim(value) for key, value in experience.items()})

        # TODO Note have to force them to be episode done

        if self.prefetch_tape.main_worker != os.getpid():
            self.prefetch_tape.update()

        device = None
        index = self.index()

        for datum in self.prefetch_tape[index][0].values():
            if hasattr(datum, 'device'):
                device = datum.device
                break

        self.prefetch_tape[index][0] = {key: add_batch_dim(datum).to(device, non_blocking=True)
                                        for key, datum in experience.items()}

    def full(self, index):
        if index > self.start_index:
            return index - self.start_index + self.batch_size > self.cap
        else:
            return self.cap - self.start_index + index + self.batch_size > self.cap

    def index(self):
        with self.index_lock:
            index = self.end_index.item()

            while self.full(index):
                pass  # Pause worker

            self.end_index[...] = (index + 1) % self.cap
            return index - 1

    def get_batch(self):
        while len(self.prefetch_tape) < self.cap:
            self.prefetch_tape.update()

        end_index = (self.start_index + self.batch_size) % self.end_index

        while end_index == self.start_index:
            # Avoid race condition when prefetch tape depleted
            end_index = (self.start_index + self.batch_size) % self.end_index

        if end_index > self.start_index:
            experiences = self.prefetch_tape[self.start_index:end_index]
        else:
            experiences = self.prefetch_tape[self.start_index:] + self.prefetch_tape[:end_index]

        # Collate
        batch = {key: torch.stack([torch.as_tensor(datum).to(self.device, non_blocking=True)
                                   for datum in [experience[0][key][...] for experience in experiences]])
                 for key in experiences[0][0]}

        return Batch(batch)
