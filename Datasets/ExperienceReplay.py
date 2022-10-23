# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import os
import random
import glob
import shutil
import atexit
import warnings
from pathlib import Path
import datetime
import io
import traceback

from omegaconf import OmegaConf

import numpy as np

import torch
from torch.utils.data import IterableDataset, Dataset
from torch.multiprocessing import Pipe

from multiprocessing import shared_memory

from torchvision.transforms import transforms


class ExperienceReplay:
    def __init__(self, batch_size, num_workers, capacity, action_spec, suite, task, offline, generate, save, load,
                 path, obs_spec=None, frame_stack=1, nstep=0, discount=1, meta_shape=(0,), transform=None):
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

        if not save:
            # Delete replay on terminate
            atexit.register(lambda p: (shutil.rmtree(p), print('Deleting replay')), self.path)

        # Data specs

        self.frame_stack = frame_stack or 1
        obs_spec.shape[0] //= self.frame_stack

        self.specs = {'obs': obs_spec, 'action': action_spec, 'meta': {'shape': meta_shape},
                      **{name: {'shape': (1,)} for name in ['reward', 'discount', 'label', 'step']}}

        # Episode traces (temporary in-RAM buffer until full episode ready to be stored)

        self.episode = {name: [] for name in self.specs}
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
            if 'Normalize' in transform:
                warnings.warn('"Normalizing" via transform. This may be redundant and dangerous if standardize=true, '
                              'which is the default.')
            # Can pass in a dict of torchvision transform names and args
            transform = transforms.Compose([getattr(transforms, t)(**transform[t]) for t in transform])

        # Future steps to compute cumulative reward from
        self.nstep = 0 if suite == 'classify' or generate else nstep

        # Parallelized experience loading, either Online or Offline - "Online" means the data size grows

        #   For now, for Offline, all data is automatically pre-loaded onto CPU RAM from hard disk before training,
        #   since RAM is faster to load from than hard disk epoch by epoch. A.K.A. training speedup, less bottleneck.
        #   We bypass Pytorch's specialized sampler indexing in order to allot disjoint slices of RAM data to each CPU.

        #   The disadvantage of CPU pre-loading is the dependency on more CPU RAM. This can be limiting in some setups.

        #       TODO: Memory-mapped hard disk loading for Offline/Online, size-adaptive w.r.t. CPU RAM loading

        #   Online also caches data on RAM, after storing to hard disk.

        #       TODO: Online can send new data directly to RAM and hard disk instead of loading it to RAM from hard disk

        # CPU workers
        self.num_workers = max(1, min(num_workers, os.cpu_count()))

        assert len(self) >= self.num_workers or not offline, f'num_workers ({self.num_workers}) ' \
                                                             f'exceeds offline replay size ({len(self)})'

        # RAM capacity per worker. Max num experiences allotted per CPU worker
        capacity = capacity // self.num_workers if capacity not in [-1, 'inf'] and not offline else np.inf

        # For sending data to workers directly
        pipes, self.pipes = zip(*[Pipe(duplex=False) for _ in range(self.num_workers)])

        self.experiences = (Offline if offline else Online)(path=self.path,
                                                            capacity=capacity,
                                                            specs=self.specs,
                                                            fetch_per=1000,
                                                            pipes=pipes,
                                                            save=save,
                                                            frame_stack=self.frame_stack,
                                                            nstep=self.nstep,
                                                            discount=discount,
                                                            transform=transform)

        # Batch loading

        self.epoch = 1

        # Offline sampler allocates per CPU worker
        batch_sampler = BatchUnionSampler(sampler=torch.utils.data.RandomSampler(self.experiences),
                                          batch_size=batch_size,
                                          num_workers=self.num_workers,
                                          shuffle=True,
                                          drop_last=False)

        self.batches = torch.utils.data.DataLoader(dataset=self.experiences,
                                                   batch_sampler=batch_sampler if offline else None,
                                                   batch_size=1 if offline else batch_size,
                                                   collate_fn=collate if offline else None,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True,
                                                   shuffle=not offline,
                                                   worker_init_fn=worker_init_fn,
                                                   persistent_workers=True)

        # Replay

        self._replay = None

    # Samples a batch of experiences, optionally includes trajectories
    def sample(self, trajectories=False):
        try:
            sample = next(self.replay)
        except StopIteration:  # Reset iterator when depleted
            self.epoch += 1
            self._replay = None
            sample = next(self.replay)
        return *sample[:6 + 4 * trajectories], *sample[10:]  # Include/exclude future trajectories

    # Allows iteration via next (e.g. batch = next(replay) )
    def __next__(self):
        return self.sample()

    # Allows iteration
    def __iter__(self):
        self._replay = iter(self.batches)
        return self.replay

    @property
    def replay(self):
        if self._replay is None:
            self._replay = iter(self.batches)  # Recreates the iterator when exhausted
        return self._replay

    # Tracks single episode "trace" in memory buffer
    def add(self, experiences=None, store=False):
        if experiences is None:
            experiences = []

        # An "episode" or part of episode of experiences
        assert isinstance(experiences, (list, tuple))

        for exp in experiences:
            for name, spec in self.specs.items():
                # Missing data
                if name not in exp:
                    exp[name] = None

                # Add batch dimension
                if np.isscalar(exp[name]) or exp[name] is None or type(exp[name]) == bool:
                    exp[name] = np.full((1, *spec['shape']), exp[name], dtype=getattr(exp[name], 'dtype', 'float32'))
                # elif len(exp[name].shape) in [0, 1, len(spec['shape'])]:
                #     exp[name].shape = (1, *spec['shape'])  # Disabled for discrete/continuous conversions

                # Expands attributes that are unique per batch (such as 'step')
                batch_size = exp.get('obs', exp['action']).shape[0]
                if 1 == exp[name].shape[0] < batch_size:
                    exp[name] = np.repeat(exp[name], batch_size, axis=0)

                # Validate consistency - disabled for discrete/continuous conversions
                # assert spec['shape'] == exp[name].shape[1:], \
                #     f'Unexpected shape for "{name}": {spec["shape"]} vs. {exp[name].shape[1:]}'

                # Add the experience
                self.episode[name].append(exp[name])

        self.episode_len += len(experiences)

        if store:
            self.store_episode()  # Stores them in file system

    # Stores episode (to file in system)
    def store_episode(self):
        if self.episode_len == 0:
            return

        for name in self.specs:
            # Concatenate into one big episode batch
            self.episode[name] = np.concatenate(self.episode[name], axis=0)

        self.episode_len = len(self.episode['obs'])

        timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        num_episodes = len(self)
        episode_name = f'{timestamp}_{num_episodes}_{self.episode_len}.npz'

        # Save episode
        with io.BytesIO() as buffer:
            np.savez_compressed(buffer, **self.episode)
            buffer.seek(0)
            with (self.path / episode_name).open('wb') as f:
                f.write(buffer.read())

        self.episode = {name: [] for name in self.specs}
        self.episode_len = 0
        self.episodes_stored += 1

    def clear(self):
        self.episode = {name: [] for name in self.specs}
        self.episode_len = 0

    # Update experiences (in workers) by IDs (experience index and worker ID) and dict like {spec: update value}
    def rewrite(self, updates, ids):
        assert isinstance(updates, dict), f'expected \'updates\' to be dict, got {type(updates)}'

        updates = {key: updates[key].detach() for key in updates}
        exp_ids, worker_ids = ids.detach().int().T

        # Send update to dedicated worker  # TODO Write to replay buffer (hard disk), but only meta spec to not corrupt
        for worker_id in torch.unique(worker_ids):
            worker = worker_ids == worker_id
            update = {key: updates[key][worker] for key in updates}
            self.pipes[worker].send((update, exp_ids[worker]))

    def __len__(self):
        return self.episodes_stored if self.offline or not self.save \
            else len(list(self.path.glob('*.npz')))


# How to initialize each worker
def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


# A CPU worker that can iteratively and efficiently build/update batches of experience in parallel (from files)
class Experiences:
    def __init__(self, path, capacity, specs, fetch_per, pipes, save, offline, frame_stack, nstep, discount, transform):

        # Dataset construction via parallel workers

        self.path = path

        self.episode_names = []
        self.episodes = dict()  # Episodes fetched on CPU

        self.experience_indices = []
        self.capacity = capacity
        self.deleted_indices = 0

        self.specs = specs

        self.fetch_per = fetch_per  # How often to fetch
        self.samples_since_last_fetch = fetch_per

        # Multiprocessing pipes
        self.pipes = pipes  # Receiving data directly from Replay

        self.save = save
        self.offline = offline

        self.frame_stack = frame_stack
        self.nstep = nstep
        self.discount = discount

        self.transform = transform

        # Load any pre-existing data file names in path
        self.data_files = sorted(path.glob('*.npz'))

        # If Offline, allocate RAM usage per CPU worker
        if self.offline:
            self.num_experiences = sum([int(episode_name.stem.split('_')[-1]) for episode_name in self.data_files])

            self.previous_idxs = None  # Verify no worker gets sent redundant batches

            # Compute data allotment per CPU worker
            worker_allotment, remainder = len(self.data_files) // len(pipes), len(self.data_files) % len(pipes)

            # Each worker gets a disjoint split of data
            self.worker_splits = [0] + [worker_allotment * (worker + 1) + remainder for worker, _ in enumerate(pipes)]

            self.allotted_idx = None

        self.initialized_worker = False

    @property
    def worker_id(self):
        return torch.utils.data.get_worker_info().id

    @property
    def num_workers(self):
        return torch.utils.data.get_worker_info().num_workers

    @property
    def pipe(self):
        return self.pipes[self.worker_id]

    # Initializes worker data allottment once they've each been created
    def init_worker(self):
        # If Offline, allocate RAM usage per CPU worker
        if self.offline:
            # Worker's starting index
            self.allotted_idx = sum([int(episode_name.stem.split('_')[-1])  # Episode len
                                     for episode_name in self.data_files[:self.worker_splits[self.worker_id]]])

            # Data per worker
            self.data_files = self.data_files[self.worker_splits[self.worker_id]:self.worker_splits[self.worker_id + 1]]

            # Load in existing data
            list(map(self.load_episode, self.data_files))

        self.initialized_worker = True

    def load_episode(self, episode_name):
        try:
            with episode_name.open('rb') as episode_file:
                episode = np.load(episode_file)
                episode = {key: episode[key] for key in episode.keys()}
        except:
            return False

        offset = self.nstep or 0
        episode_len = len(episode['obs']) - offset
        episode = {name: episode.get(name, np.full((episode_len + 1, *spec['shape']), np.NaN))
                   for name, spec in self.specs.items()}  # TODO Shared memory dict somehow for Offline

        episode['id'] = len(self.experience_indices)
        self.experience_indices += list(enumerate([episode_name] * episode_len))  # TODO Shared memory list

        self.episodes[episode_name] = episode  # TODO Shared memory dict of dicts (double keys?) somehow for Offline

        # TODO Can have each worker load partially then overwrite experience_indices with a shared one or redundant

        # Deleting experiences upon overfill
        while episode_len + len(self) - self.deleted_indices > self.capacity:
            early_episode_name = self.episode_names.pop(0)
            early_episode = self.episodes.pop(early_episode_name)
            early_episode_len = len(early_episode['obs']) - offset
            self.deleted_indices += early_episode_len  # To derive a consistent experience index even as data deleted
            if not self.save:
                # Deletes early episode file
                early_episode_name.unlink(missing_ok=True)

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

            self.episode_names.append(episode_name)
            self.episode_names.sort()

            if not self.save:
                episode_name.unlink(missing_ok=True)  # Deletes file

    # Can update/write data based on piped update specs
    def worker_fetch_updates(self):
        while self.pipe.poll():
            updates, exp_ids = self.pipe.recv()

            # Iterate through each update spec
            for key in updates:
                for update, exp_id in zip(updates[key], exp_ids):
                    # Get corresponding experience and episode
                    idx, episode_name = self.experience_indices[exp_id]

                    # Update experience in replay
                    if episode_name in self.episodes:
                        self.episodes[episode_name][key][idx] = update.numpy()

                    # TODO Update experience's "meta" spec in hard disk (memory-mapping needs to be implemented first)

    def sample(self, episode_names, metrics=None):
        episode_name = random.choice(episode_names)  # Uniform sampling of experiences
        return episode_name

    # N-step cumulative discounted rewards
    def process(self, episode, idx=None):
        offset = self.nstep or 0
        episode_len = len(episode['obs']) - offset
        if idx is None:
            idx = np.random.randint(episode_len)

        # Frame stack
        def frame_stack(traj_o, idx):
            frames = traj_o[max([0, idx + 1 - self.frame_stack]):idx + 1]
            for _ in range(self.frame_stack - idx - 1):
                frames = np.concatenate([traj_o[:1], frames], 0)
            frames = frames.reshape(frames.shape[1] * self.frame_stack, *frames.shape[2:])
            return frames

        # Present
        obs = frame_stack(episode['obs'], idx)
        label = episode['label'][idx]
        step = episode['step'][idx]

        exp_id, worker_id = episode['id'] + idx, self.worker_id
        ids = np.array([exp_id, worker_id])

        meta = episode['meta'][idx]  # Agent-writable Metadata

        # Future
        if self.nstep:
            # Transition
            action = episode['action'][idx + 1]
            next_obs = frame_stack(episode['obs'], idx + self.nstep)

            # Trajectory
            traj_o = np.concatenate([episode['obs'][max(0, idx - i):max(idx + self.nstep + 1 - i, self.nstep + 1)]
                                     for i in range(self.frame_stack - 1, -1, -1)], 1)  # Frame_stack
            traj_a = episode['action'][idx + 1:idx + self.nstep + 1]
            traj_r = episode['reward'][idx + 1:idx + self.nstep + 1]
            traj_l = episode['label'][idx:idx + self.nstep + 1]

            # Cumulative discounted reward
            discounts = self.discount ** np.arange(self.nstep + 1)
            reward = np.dot(discounts[:-1], traj_r)
            discount = discounts[-1:]
        else:
            action, reward = episode['action'][idx], episode['reward'][idx]

            next_obs = traj_o = traj_a = traj_r = traj_l = np.full((0,), np.NaN)
            discount = np.array([1.0])

        # Transform
        if self.transform is not None:
            obs = self.transform(torch.as_tensor(obs))

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
            idx, episode_name = self.experience_indices[idx + self.deleted_indices]

        episode = self.episodes[episode_name]

        return self.process(episode, idx)  # Process episode into a compact experience

    def __len__(self):
        return (self.num_experiences if self.offline
                else len(self.experience_indices)) - self.deleted_indices


# Loads Experiences with an Iterable Dataset
class Online(Experiences, IterableDataset):
    def __init__(self, path, capacity, specs, fetch_per, pipes, save, frame_stack, nstep=0, discount=1, transform=None):
        super().__init__(path, capacity, specs, fetch_per, pipes, save, False, frame_stack, nstep, discount, transform)

    def __iter__(self):
        # Keep fetching, sampling, and building batches
        while True:
            yield self.fetch_sample_process()  # Yields a single experience


a = [19067, 27997, 18851, 29292, 17512, 17821, 16187, 16862, 24153, 30448, 28020, 16039, 29612, 23577, 30138, 29395, 22847, 18060, 21397, 25785, 25051, 22536, 22250, 22793, 26358, 16185, 25874, 26463, 25942, 24892, 26864, 21490, 17403, 27782, 18002, 23837, 24699, 27636, 22291, 25882, 30443, 24217, 21887, 23842, 18551, 29900, 16005, 29215, 23228, 21973, 28423, 21560, 29240, 22798, 27449, 25726, 26236, 27028, 25554, 20021, 23828, 25576, 17136, 26031, 27993, 30340, 25008, 27970, 19230, 24468, 17434, 29583, 29353, 27991, 29452, 20161, 26514]
b = [57390, 55859, 55717, 56856, 57510, 57981, 49777, 59506, 59372, 57002, 58611, 50195, 55596, 54893, 54101, 46476, 52918, 46831, 58480, 55312, 54806, 52443, 52777, 53619, 46163, 54607, 54877, 57719, 59926, 54456, 47369, 52075, 45523, 59091, 46167, 53062, 47985, 46925, 52803, 52937, 53250, 54333, 57362, 49077, 57939, 46677, 46436, 45352, 53620, 55919, 50878, 45903, 59399, 50570, 45737, 49948, 51438, 58885, 45831, 48516, 46863, 53379, 57130, 46998, 48515, 49758, 46990, 47587, 53490, 50918]
c = [3203, 14353, 6119, 14963, 9978, 4909, 465, 14017, 5109, 15032, 3334, 7848, 15585, 10238, 2372, 4621, 6600, 2708, 7705, 3521, 10080, 4184, 650, 6438, 13406, 14221, 2624, 7627, 2469, 925, 3246, 10456, 12720, 9979, 593, 778, 8229, 5741, 12822, 932, 1064, 7806, 7422, 9200, 2935, 1974, 10463, 14381, 9764, 12667, 3458, 7891, 7114, 7191, 10050, 5642, 12484, 11798, 3694, 8638, 6903, 8612, 452, 13211, 10854, 1651, 12701]
d = [38651, 37660, 31745, 44900, 41559, 39650, 33384, 32152, 36599, 34507, 30936, 42707, 41859, 38818, 37634, 35191, 42720, 36705, 40004, 32002, 30865, 34265, 38593, 31522, 33347, 36921, 35269, 30821, 44579, 31781, 37268, 35106, 31619, 44503, 36438, 32714, 41302, 33564, 33790, 35940, 41429, 41015, 39260, 42467, 30972, 34673, 42776, 34625, 36159, 31583, 38692, 34903, 35988, 37083, 44818, 35899, 42053, 35196, 39818, 40954, 31140, 40982, 40089, 31613, 45153]


# Loads Experiences with a Map Style Dataset
class Offline(Experiences, Dataset):
    def __init__(self, path, capacity, specs, fetch_per, pipes, save, frame_stack, nstep=0, discount=1, transform=None):
        super().__init__(path, capacity, specs, fetch_per, pipes, save, True, frame_stack, nstep, discount, transform)

    def __getitem__(self, idxs):
        if self.worker_id == 2:
            print(idxs[0])
        assert idxs != self.previous_idxs, f'BatchUnionSampler failed to allocate index uniquely to each CPU worker.'
        self.previous_idxs = idxs  # TODO Delete / / Indeed, it fails

        if not self.initialized_worker:

            # Can replace with universal hard disk to worker index mapping and using the auto-fetching
            self.init_worker()

            # print([idx for idx in idxs[0] if self.allotted_idx <= idx < self.allotted_idx + len(self.experience_indices)])
            # print(len(idxs), self.worker_id,
            #       len([idx for idx in idxs[0] if self.allotted_idx <= idx < self.allotted_idx + len(self.experience_indices)]))

        # Each worker retrieves from a pre-allotted share
        return [self.fetch_sample_process(idx - self.allotted_idx)
                for idx in idxs[0] if self.allotted_idx <= idx < self.allotted_idx + len(self.experience_indices)]


# For Offline, CPU workers distribute indices (sub-batches) amongst themselves according to their disjoint RAM allotment
class BatchUnionSampler(torch.utils.data.BatchSampler):
    def __init__(self, sampler, batch_size, num_workers, shuffle=False, drop_last=False):
        super().__init__(sampler, batch_size, drop_last)

        self.num_workers, self.shuffle = num_workers, shuffle

    # Each CPU gets a disjoint slice of RAM data instead of a wasteful replica
    def __iter__(self):
        batches = list(super().__iter__())

        if self.shuffle:
            random.shuffle(batches)

        # Send all batch indices (union) to each CPU worker
        for batch in batches:
            yield [[batch]] * self.num_workers


# Collates sub-batches for Offline
def collate(batch):
    return torch.utils.data.default_collate([elem for sub_batch in batch for elem in sub_batch])


class SharedMemory(dict):
    def __setitem__(self, key, value):
        assert isinstance(value, dict), 'Shared Memory assumed to consist of dicts'

        self[key] = {}

        for key_, value_ in value.items():
            if isinstance(value, np.ndarray):
                shared_memory.SharedMemory(create=False, name=key + key_, size=value_.nbytes)  # Set create exception
                self[key][key_] = None
            else:
                self[key][key_] = value

    def __getitem__(self, key):
        value = self[key]
        for key_, value_ in value.items():
            if value_ is None:
                self[key][key_] = shared_memory.SharedMemory(create=False, name=key + key_)

        return self[key]
