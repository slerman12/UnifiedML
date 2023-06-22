# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""
A new Replay memory. Programmed by Sam Lerman.
"""

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
from World.Dataset import load_dataset, datums_as_batch, get_dataset_path, Transform, worker_init_fn
from Hyperparams.minihydra import instantiate, Args, added_modules


class Replay:
    def __init__(self, path='Replay/', batch_size=1, device='cpu', num_workers=0, offline=True, stream=False,
                 gpu_capacity=0, pinned_capacity=0, tensor_ram_capacity=1e6, ram_capacity=0, hd_capacity=inf,
                 save=False, mem_size=None, fetch_per=1,
                 prefetch_factor=3, pin_memory=False, pin_device_memory=False, shuffle=True,
                 dataset=None, transform=None, frame_stack=1, nstep=None, discount=1, meta_shape=(0,)):

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
                             tensor_ram_capacity=tensor_ram_capacity,
                             ram_capacity=ram_capacity,
                             hd_capacity=hd_capacity)

        self.add_lock = Lock()  # For adding to memory in concurrency

        dataset_config = dataset
        card = Args({'_target_': dataset_config}) if isinstance(dataset_config, str) else dataset_config
        # TODO Mark that training if not marked, and use card universally, and don't include str handling

        if dataset_config is not None and dataset_config._target_ is not None:
            # Pytorch Dataset or Memory path
            dataset = load_dataset('World/ReplayBuffer/Offline/', dataset_config)

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
                # TODO Apply dataset.transform on this
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

        # self.action_spec = {'shape': (1,),
        #                     'discrete_bins': len(classes),
        #                     'low': 0,
        #                     'high': len(classes) - 1,
        #                     'discrete': True}
        #
        # self.obs_spec = {'shape': obs_shape,
        #                  'mean': mean,
        #                  'stddev': stddev,
        #                  'low': low,
        #                  'high': high}

        # Unique classes in dataset - warning: treats multi-label as single-label for now
        # # TODO Save/Only do once - debug speech command on Macula
        # classes = subset if subset is not None \
        #     else range(len(getattr(dataset, 'classes'))) if hasattr(dataset, 'classes') \
        #     else dataset.class_to_idx.keys() if hasattr(dataset, 'class_to_idx') \
        #     else [print(f'Identifying unique {"train" if train else "eval"} classes... '
        #                 f'This can take some time for large datasets.'),
        #           sorted(list(set(str(exp[1]) for exp in dataset)))][1]
        #
        # # Can select a subset of classes
        # if subset:
        #     task += '_Classes_' + '_'.join(map(str, classes))
        #     print(f'Selecting subset of classes from dataset... This can take some time for large datasets.')
        #     dataset = ClassSubset(dataset, classes)
        #
        # # Map unique classes to integers
        # dataset = ClassToIdx(dataset, classes)
        #
        # # Transform inputs
        # transform = instantiate(transform)
        # if transform:
        #     task += '_Transformed'  # Note: These name changes only apply to replay buffer and not benchmarking yet
        # dataset = Transform(dataset, transform)



        # """Create Replay and compute stats"""
        #
        # replay_path = Path(f'./Datasets/ReplayBuffer/Classify/{task}_Buffer')
        #
        # stats_path = glob.glob(f'./Datasets/ReplayBuffer/Classify/{task}_Stats*')
        #
        # # Parallelism-protection, but note that clashes may still occur in multi-process dataset creation
        # if replay_path.exists() and not len(stats_path):
        #     warnings.warn(f'Incomplete or corrupted replay. If you launched multiple processes, then another one may be'
        #                   f' creating the replay still, in which case, just wait. Otherwise, kill this process (ctrl-c)'
        #                   f' and delete the existing path (`rm -r <Path>`) and try again to re-create.\nPath: '
        #                   f'{colored(replay_path, "green")}\n{"Also: " + stats_path[0] if len(stats_path) else ""}'
        #                   f'{colored("Wait (do nothing)", "yellow")} '
        #                   f'{colored("or kill (ctrl-c), delete path (rm -r <Path>) and try again.", "red")}')
        #     while not len(stats_path):
        #         sleep(10)  # Wait 10 sec
        #
        #         stats_path = glob.glob(f'./Datasets/ReplayBuffer/Classify/{task}_Stats*')
        #
        # # Offline and generate don't use training rollouts (unless streaming)
        # if (offline or generate) and not (stream or train or replay_path.exists()):
        #     # But still need to create training replay & compute stats
        #     Classify(dataset_, None, task_, True, offline, generate, stream, batch_size, num_workers, subset, None,
        #              None, seed, transform, **kwargs)
        #
        # # Create replay
        # if train and (offline or generate) and not (replay_path.exists() or stream):
        #     self.create_replay(replay_path)
        #
        # stats_path = glob.glob(f'./Datasets/ReplayBuffer/Classify/{task}_Stats*')
        #
        # # Compute stats
        # mean, stddev, low_, high_ = map(json.loads, open(stats_path[0]).readline().split('_')) if len(stats_path) \
        #     else self.compute_stats(f'./Datasets/ReplayBuffer/Classify/{task}') if train and not stream else (None,) * 4
        # low, high = low_ if low is None else low, high_ if high is None else high

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
                                                   pin_memory=pin_memory and 'cuda' in device and not pin_device_memory,
                                                   prefetch_factor=prefetch_factor if num_workers else 2,
                                                   shuffle=shuffle and offline,
                                                   worker_init_fn=worker_init_fn,
                                                   persistent_workers=bool(num_workers))

        # Replay

        self.replay = iter(self)

    # Allows iteration via "next" (e.g. batch = next(replay))
    def __next__(self):
        if self.stream:
            # Environment streaming
            return self.stream
        else:
            # Replay sampling
            try:
                sample = next(self.replay)
            except StopIteration:
                self.epoch += 1
                self.replay = iter(self)
                sample = next(self.replay)

            return Batch({key: value.to(self.device, non_blocking=True) for key, value in sample.items()})

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

    def writable_tape(self, batch, ind, step):
        assert isinstance(batch, (dict, Batch)), f'expected \'batch\' to be dict or Batch, got {type(batch)}.'
        self.memory.writable_tape(batch, ind, step)

    def __len__(self):
        # Infinite if stream, else num episodes in Memory
        return inf if self.stream else len(self.memory)


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

        if len(episode) < self.nstep + 1:  # TODO support <nstep
            return self.sample(_index, update=True)

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
            experience['next_obs'] = frame_stack(episode, 'obs', step + self.nstep)

            traj_r = torch.as_tensor([experience.reward
                                      for experience in episode[step + 1:step + self.nstep + 1]])

            # Trajectory TODO
            if self.trajectory_flag:
                experience['traj_r'] = traj_r
                traj_o = np.concatenate([episode['obs'][max(0, idx - i):max(idx + self.nstep + 1 - i, self.nstep + 1)]
                                         for i in range(self.frame_stack - 1, -1, -1)], 1)  # Frame_stack
                traj_a = episode['action'][idx + 1:idx + self.nstep + 1]
                if 'label' in experience:
                    traj_l = episode['label'][idx:idx + self.nstep + 1]

            # Cumulative discounted reward
            discounts = self.discount ** np.arange(self.nstep + 1)
            experience.reward = np.dot(discounts[:-1], traj_r).astype('float32')
            experience['discount'] = discounts[-1].astype('float32')
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
