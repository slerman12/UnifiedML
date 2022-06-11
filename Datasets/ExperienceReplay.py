# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import asyncio
import multiprocessing
import os
import random
import glob
import shutil
import atexit
import threading
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
import datetime
import io
import traceback

from omegaconf import OmegaConf

import numpy as np

import torch
from torch.utils.data import IterableDataset, Dataset

from torchvision.transforms import transforms
import h5py


class ExperienceReplay:
    def __init__(self, batch_size, num_workers, capacity, action_spec, suite, task, offline, generate, save, load, path,
                 obs_spec=None, nstep=0, discount=1, meta_shape=None, transform=None):
        # Path and loading

        exists = glob.glob(path + '*/')

        offline = offline or generate

        if load or offline:
            if suite == 'classify':
                standard = f'./Datasets/ReplayBuffer/Classify/{task}_Buffer'
                if len(exists) == 0:
                    exists = [standard + '/']
                    print('All data loaded. Training of classifier underway.')
                else:
                    if path != standard:
                        warnings.warn(f'Loading a saved replay of a classify task from a previous online session.'
                                      f'For the standard offline dataset, set replay.path="{standard}" '  # If exists
                                      f'or delete the saved buffer in {path}.')
            assert len(exists) > 0, f'No existing replay buffer found in path: {path}'
            self.path = Path(sorted(exists)[-1])
            save = offline or save
        else:
            self.path = Path(path + '_' + str(datetime.datetime.now()))
            self.path.mkdir(exist_ok=True, parents=True)

        # Directory for sending updates to replay workers
        (self.path / 'Updates').mkdir(exist_ok=True, parents=True)

        # Delete any pre-existing ones
        update_names = (self.path / 'Updates').glob('*.npz')
        for update_name in update_names:
            update_name.unlink(missing_ok=True)  # Deletes file

        if not save:
            # Delete replay on terminate
            atexit.register(lambda p: (shutil.rmtree(p), print('Deleting replay')), self.path)

        # Data specs

        # todo obs_shape, action_shape, meta_shape, no specs
        # self.specs = {'obs': obs_shape <or (1,)?>, 'action': action_shape, 'meta': meta_shape}
        # self.specs.update({name: (0,) for name in ['reward', 'discount', 'label', 'step']})

        if obs_spec is None:
            obs_spec = {'name': 'observation', 'shape': (1,)}  # todo when is obs-spec none?

        self.specs = (obs_spec, action_spec, *[{'name': name, 'shape': (1,)}
                                               for name in ['reward', 'discount', 'label', 'step']],
                      {'name': 'meta', 'shape': meta_shape})

        # Episode traces (temporary in-RAM buffer until full episode ready to be stored)

        self.episode = {spec['name']: [] for spec in self.specs}
        self.episode_len = 0
        self.episodes_stored = len(list(self.path.glob('*.npz')))
        self.save = save
        self.offline = offline

        # Data transform

        if transform is not None:
            if isinstance(transform, str):
                transform = OmegaConf.create(transform)
            if 'RandomCrop' in transform and 'size' not in transform['RandomCrop']:
                transform['RandomCrop']['size'] = obs_spec['shape'][-2:]
            # Can pass in a dict of torchvision transform names and args
            transform = transforms.Compose([getattr(transforms, t)(**transform[t]) for t in transform])

        # Parallelized experience loading, either online or offline

        self.nstep = nstep

        self.num_workers = max(1, min(num_workers, os.cpu_count()))

        assert len(self) >= self.num_workers or not offline, f'num_workers ({self.num_workers}) ' \
                                                             f'exceeds offline replay size ({len(self)})'

        capacity = capacity // self.num_workers if capacity and not offline \
            else np.inf

        self.queue = multiprocessing.Queue()  # TODO use pipe

        self.experiences = (Offline if offline else Online)(path=self.path,
                                                            queue=self.queue,
                                                            capacity=capacity,
                                                            specs=self.specs,
                                                            fetch_per=1000,
                                                            save=save,
                                                            nstep=nstep,
                                                            discount=discount,
                                                            transform=transform)

        # Batch loading

        self.epoch = 1

        self.batches = torch.utils.data.DataLoader(dataset=self.experiences,
                                                   batch_size=batch_size,
                                                   shuffle=offline,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True,
                                                   worker_init_fn=worker_init_fn,
                                                   persistent_workers=True)
        # Replay

        self._replay = None

    # Returns a batch of experiences
    def sample(self):
        return next(self)  # Can iterate via next

    # Allows iteration
    def __next__(self):
        try:
            return next(self.replay)
        except StopIteration:
            self.epoch += 1
            self._replay = None
            return next(self)

    # Allows iteration
    def __iter__(self):
        self._replay = iter(self.batches)
        return self.replay

    @property
    def replay(self):
        if self._replay is None:
            self._replay = iter(self.batches)
        return self._replay

    # Tracks single episode "trace" in memory buffer
    def add(self, experiences=None, store=False):
        if experiences is None:
            experiences = []

        # An "episode" or part of episode of experiences
        assert isinstance(experiences, (list, tuple))

        for exp in experiences:
            for spec in self.specs:
                # Missing data
                if not hasattr(exp, spec['name']):
                    setattr(exp, spec['name'], None)

                # Add batch dimension
                if np.isscalar(exp[spec['name']]) or exp[spec['name']] is None:
                    exp[spec['name']] = np.full((1,) + tuple(spec['shape']), exp[spec['name']], 'float32')
                if len(exp[spec['name']].shape) == len(spec['shape']):
                    exp[spec['name']] = np.expand_dims(exp[spec['name']], 0)

                # Expands attributes that are unique per batch (such as 'step')
                batch_size = getattr(exp, 'observation', getattr(exp, 'action')).shape[0]
                if 1 == exp[spec['name']].shape[0] < batch_size:
                    exp[spec['name']] = np.repeat(exp[spec['name']], batch_size, axis=0)

                # Validate consistency
                assert spec['shape'] == exp[spec['name']].shape[1:], \
                    f'Unexpected {spec["name"]} shape: {spec["shape"]} vs. {exp[spec["name"]].shape}'

                # Add the experience
                self.episode[spec['name']].append(exp[spec['name']])

        self.episode_len += len(experiences)

        if store:
            self.store_episode()  # Stores them in file system

    # Stores episode (to file in system)
    def store_episode(self):
        if self.episode_len == 0:
            return

        for spec in self.specs:
            # Concatenate into one big episode batch
            self.episode[spec['name']] = np.concatenate(self.episode[spec['name']], axis=0)

        self.episode_len = len(self.episode['observation'])

        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        num_episodes = len(self)
        episode_name = f'{timestamp}_{num_episodes}_{self.episode_len}.npz'

        # Save episode
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **self.episode)
            buffer.seek(0)
            with (self.path / episode_name).open('wb') as f:
                f.write(buffer.read())

        self.episode = {spec['name']: [] for spec in self.specs}
        self.episode_len = 0
        self.episodes_stored += 1

    # Update experiences (in workers) by IDs (experience index and worker ID) and dict like {spec: update value}
    def rewrite(self, updates, ids):
        assert isinstance(updates, dict), f'expected \'updates\' to be dict, got {type(updates)}'

        updates = {key: updates[key].detach().cpu().numpy() for key in updates}

        # Store into replay buffer
        for i, (exp_id, worker_id) in enumerate(zip(*ids.int().T)):
            # In the offline setting, each worker has a copy of all the data
            for worker in range(self.num_workers):

                timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
                update_name = f'{exp_id}_{worker if self.offline else worker_id}_{timestamp}.npz'
                update = {key: updates[key][i] for key in updates}

                # Send update to workers
                self.queue.put(update)
                # with io.BytesIO() as buffer:
                #     np.savez_compressed(buffer, **update)
                #     buffer.seek(0)
                #     with (self.path / 'Updates' / update_name).open('wb') as f:
                #         f.write(buffer.read())
                # file = h5py.File('arrays.h5', 'w', dtype=data.dtype)
                # file.create_dataset(update_name, data=update)

                if not self.offline:
                    break

    # Update experiences (in workers) by IDs (experience index and worker ID) and dict like {spec: update value}
    # def rewrite(self, updates, ids):
    #     assert isinstance(updates, dict), f'expected \'updates\' to be dict, got {type(updates)}'
    #
    #     updates = {key: updates[key].detach().numpy() for key in updates}
    #
    #     threading.Thread(target=store, args=(updates, ids, self.num_workers, self.offline, self.path)).start()

    # multiprocessing.Process(target=store, args=(updates, ids, self.num_workers, self.offline, self.path)).start()

    # store(*(updates, ids, self.num_workers, self.offline, self.path))

    # Update experiences (in workers) by ID (experience, worker ID) and dict like {spec: update value}
    # def rewrite(self, updates, ids):
    #     assert isinstance(updates, dict), f'expected \'updates\' to be dict, got {type(updates)}'
    #
    #     exp_ids, worker_ids = ids.T
    #     updates['exp_ids'] = exp_ids
    #
    #     # Store into replay buffer per worker
    #     for worker_id in torch.unique(worker_ids, sorted=False):
    #         # In the offline setting, each worker has a copy of all the data
    #         per_worker = self.offline or (ids[:, 1] == worker_id)  # Otherwise each worker has dedicated data
    #
    #         update = {key: updates[key][per_worker].cpu().numpy() for key in updates}
    #
    #         timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    #         update_name = f'{worker_id}_{timestamp}.npz'
    #
    #         # Send update to workers
    #         with io.BytesIO() as buffer:
    #             np.savez_compressed(buffer, update)
    #             buffer.seek(0)
    #             with (self.path / 'Updates' / update_name).open('wb') as f:
    #                 f.write(buffer.read())

    def __len__(self):
        return self.episodes_stored if self.offline or not self.save \
            else len(list(self.path.glob('*.npz')))


# How to initialize each worker
def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


# def store(updates, ids, num_workers, offline, path):
#     for i, (exp_id, worker_id) in enumerate(zip(*ids.int().T)):
#         # In the offline setting, each worker has a copy of all the data
#         for worker in range(num_workers):
#
#             timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
#             update_name = f'{exp_id}_{worker if offline else worker_id}_{timestamp}.npz'
#             update = {key: updates[key][i] for key in updates}
#
#             # Send update to workers
#             with io.BytesIO() as buffer:
#                 np.savez_compressed(buffer, **update)
#                 buffer.seek(0)
#                 with (path / 'Updates' / update_name).open('wb') as f:
#                     f.write(buffer.read())
#
#             if not offline:
#                 break


# A CPU worker that can iteratively and efficiently build/update batches of experience in parallel (from files)
class Experiences:
    def __init__(self, path, queue, capacity, specs, fetch_per, save, offline=True, nstep=0, discount=1, transform=None):

        # Dataset construction via parallel workers

        self.path = path

        self.queue = queue

        self.episode_names = []
        self.episodes = dict()

        self.experience_indices = []
        self.capacity = capacity
        self.deleted_indices = 0

        self.specs = specs

        self.fetch_per = fetch_per  # How often to fetch
        self.samples_since_last_fetch = fetch_per

        self.save = save
        self.offline = offline

        self.nstep = nstep
        self.discount = discount

        self.transform = transform

        # Load in existing data
        _ = list(map(self.load_episode, sorted(path.glob('*.npz'))))

    @property
    def worker_id(self):
        return torch.utils.data.get_worker_info().id

    @property
    def num_workers(self):
        return torch.utils.data.get_worker_info().num_workers

    def load_episode(self, episode_name):
        try:
            with episode_name.open('rb') as episode_file:
                episode = np.load(episode_file)
                episode = {key: episode[key] for key in episode.keys()}
        except:
            return False

        offset = self.nstep or 1
        episode_len = len(episode['observation']) - offset
        episode = {spec['name']: episode.get(spec['name'], np.full((episode_len + 1, *spec['shape']), np.NaN))
                   for spec in self.specs}

        # Deleting experiences upon overfill
        while episode_len + len(self) > self.capacity:
            early_episode_name = self.episode_names.pop(0)
            early_episode = self.episodes.pop(early_episode_name)
            early_episode_len = len(early_episode['observation']) - offset
            self.experience_indices = self.experience_indices[early_episode_len:]
            self.deleted_indices += early_episode_len  # To derive a consistent experience index even as data deleted
            if not self.save:
                # Deletes early episode file
                early_episode_name.unlink(missing_ok=True)

        episode['id'] = len(self.experience_indices) + self.deleted_indices  # IDs remain unique even if experiences deleted

        self.episode_names.append(episode_name)
        self.episode_names.sort()
        self.episodes[episode_name] = episode
        self.experience_indices += list(enumerate([episode_name] * episode_len))

        if not self.save:
            episode_name.unlink(missing_ok=True)  # Deletes file

        return True

    # Populates workers with new data
    def worker_fetch_episodes(self):
        if self.samples_since_last_fetch < self.fetch_per:
            return

        self.samples_since_last_fetch = 0

        episode_names = sorted(self.path.glob('*.npz'), reverse=True)  # Episodes
        num_fetched = 0
        # Find new episodes
        for episode_name in episode_names:
            episode_idx, episode_len = [int(x) for x in episode_name.stem.split('_')[1:]]
            if episode_idx % self.num_workers != self.worker_id:  # Each worker stores their own dedicated data
                continue
            if episode_name in self.episodes.keys():  # Don't store redundantly
                break
            # if num_fetched + episode_len > self.capacity:  # Don't overfill  (This is already accounted for)
            #     break
            num_fetched += episode_len
            if not self.load_episode(episode_name):
                break  # Resolve conflicts

    # Workers can update/write-to their own data based on file-stored update specs
    def worker_fetch_updates(self):
        update_names = (self.path / 'Updates').glob('*.npz')

        if self.worker_id == 2:
            while not self.queue.empty():
                bla = self.queue.get()

        # Fetch update specs
        for i, update_name in enumerate(update_names):
            exp_id, worker_id = [int(ids) for ids in update_name.stem.split('_')[:2]]

            if worker_id != self.worker_id:  # Each worker updates own dedicated data
                continue

            # Get corresponding experience and episode
            idx, episode_name = self.experience_indices[exp_id - self.deleted_indices]

            try:
                with update_name.open('rb') as update_file:
                    update = np.load(update_file)

                    # Iterate through each update spec
                    for key in update.keys():
                        # Update experience in replay
                        self.episodes[episode_name][key][idx] = update[key]
                update_name.unlink(missing_ok=True)  # Delete update spec when stored
            except:
                continue

    # Workers can update/write-to their own data based on file-stored update specs
    # def worker_fetch_updates(self):
    #     update_names = (self.path / 'Updates').glob('*.npz')
    #
    #     # Fetch update specs
    #     for update_name in update_names:
    #         worker_id = int(update_name.stem.split('_')[0])
    #
    #         if worker_id != self.worker_id:  # Each worker updates own dedicated data
    #             continue
    #
    #         try:
    #             with update_name.open('rb') as update_file:
    #                 update = np.load(update_file)
    #                 update = {key: update[key] for key in update.keys()}
    #         except:
    #             continue
    #
    #         for i, exp_id in enumerate(update.pop('exp_ids')):
    #             # Get corresponding experience and episode
    #             idx, episode_name = self.experience_indices[exp_id - self.deleted_indices]
    #
    #             # Iterate through each update spec
    #             for key in update.keys():
    #                 # Update experience in replay
    #                 self.episodes[episode_name][key][idx] = update[key][i]
    #             update_name.unlink(missing_ok=True)  # Delete update spec when stored

    def sample(self, episode_names, metrics=None):
        episode_name = random.choice(episode_names)  # Uniform sampling of experiences
        return episode_name

    # N-step cumulative discounted rewards
    def process(self, episode, idx=None):
        offset = self.nstep or 1
        episode_len = len(episode['observation']) - offset
        if idx is None:
            idx = np.random.randint(episode_len)

        # Transition
        obs = episode['observation'][idx]
        action = episode['action'][idx + 1]
        next_obs = episode['observation'][idx + self.nstep]
        label = episode['label'][idx].squeeze()
        step = episode['step'][idx]

        exp_id, worker_id = episode['id'] + idx, self.worker_id
        ids = np.array([exp_id, worker_id])

        meta = episode['meta'][idx]  # Agent-writable Metadata

        # Trajectory
        if self.nstep:
            traj_o = episode['observation'][idx:idx + self.nstep + 1]
            traj_a = episode['action'][idx + 1:idx + self.nstep + 1]  # 1 len smaller than traj_o
            traj_r = episode['reward'][idx + 1:idx + self.nstep + 1]  # 1 len smaller than traj_o
            traj_l = episode['label'][idx:idx + self.nstep + 1]

            # Compute cumulative discounted reward TODO store cache
            discounts = self.discount ** np.arange(self.nstep + 1)
            reward = np.dot(discounts[:-1], traj_r)
            discount = discounts[-1:]
        else:
            traj_o = traj_a = traj_r = traj_l = reward = np.array([np.NaN])
            discount = np.array([1.0])

        # Compute cumulative discounted reward
        # reward = np.array([np.NaN])
        # discount = np.array([1.0])
        # for i in range(1, self.nstep + 1):
        #     if episode['reward'][idx + i] != np.NaN:
        #         step_reward = episode['reward'][idx + i]
        #         if np.isnan(reward):
        #             reward = np.zeros(1)
        #         reward += discount * step_reward
        #         discount *= episode['discount'][idx + i] * self.discount

        # Transform
        if self.transform is not None:  # TODO audio
            obs = self.transform(torch.as_tensor(obs).div(255)) * 255

        return obs, action, reward, discount, next_obs, label, traj_o, traj_a, traj_r, traj_l, step, ids, meta

    def fetch_sample_process(self, idx=None):
        try:
            # Populate workers with up-to-date data
            if not self.offline:
                self.worker_fetch_episodes()
            self.worker_fetch_updates()
        except:
            traceback.print_exc()

        self.samples_since_last_fetch += 1

        # Sample or index an experience
        if idx is None:
            episode_name = self.sample(self.episode_names)
        else:
            idx, episode_name = self.experience_indices[idx]

        episode = self.episodes[episode_name]

        return self.process(episode, idx)  # Process episode into a compact experience

    def __len__(self):
        return len(self.experience_indices)


# Loads Experiences with an Iterable Dataset
class Online(Experiences, IterableDataset):
    def __init__(self, path, queue, capacity, specs, fetch_per, save, nstep=0, discount=1, transform=None):
        super().__init__(path, queue, capacity, specs, fetch_per, save, False, nstep, discount, transform)

    def __iter__(self):
        # Keep fetching, sampling, and building batches
        while True:
            yield self.fetch_sample_process()  # Yields a single experience


# Loads Experiences with a standard Dataset
class Offline(Experiences, Dataset):
    def __init__(self, path, queue, capacity, specs, fetch_per, save, nstep=0, discount=1, transform=None):
        super().__init__(path, queue, capacity, specs, fetch_per, save, True, nstep, discount, transform)

    def __getitem__(self, idx):
        return self.fetch_sample_process(idx)  # Get single experience by index
