# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import os
import random
import glob
import shutil
import atexit
import uuid
import warnings
import resource
from pathlib import Path
import datetime
import io
import traceback
from time import sleep

from omegaconf import OmegaConf

import numpy as np

from multiprocessing.shared_memory import SharedMemory, ShareableList
from multiprocessing import resource_tracker

import torch
from torch.utils.data import IterableDataset, Dataset
from torch.multiprocessing import Pipe

from torchvision.transforms import transforms


class ExperienceReplay:
    def __init__(self, batch_size, num_workers, capacity, suite, task, offline, generate, stream, save, load, path, env,
                 obs_spec, action_spec, frame_stack=1, nstep=0, discount=1, meta_shape=(0,), transform=None):
        # Path and loading

        exists = glob.glob(path + '*/')

        offline = offline or generate

        if load or offline:
            if suite == 'classify':
                if env.subset:
                    task += '_Classes_' + '_'.join(map(str, env.subset))  # Subset of classes
                if getattr(env.transform, '_target_', env.transform) is not None:
                    task += '_Transformed'  # Pre-transformed by environment (fixed transformations)
                standard = f'./Datasets/ReplayBuffer/Classify/{task}_Buffer'
                if len(exists) == 0:
                    exists = [standard + '/']
                    print('All data loaded. Training of classifier underway.')
                else:
                    if path != standard:
                        warnings.warn(f'Loading a saved replay of a classify task from a previous online session. '
                                      f'For the standard offline dataset, set replay.path="{standard}" '  # If exists
                                      f'or delete the saved buffer in {path}.')
            assert len(exists) > 0, f'\nNo existing replay buffer found in path: {path}.\ngenerate=true, ' \
                                    f'offline=true, & replay.load=true all assume the presence of a saved replay ' \
                                    f'buffer. \nTry replay.save=true first, then you can try again with one ' \
                                    f'of those 3, \nor set those to false, or choose a different path via replay.path=.'
            self.path = Path(sorted(exists)[-1])
            save = offline or save
        elif not stream:
            self.path = Path(path + '_' + str(datetime.datetime.now()))
            self.path.mkdir(exist_ok=True, parents=True)

        if not save and hasattr(self, 'path'):
            # Delete replay on terminate
            atexit.register(lambda p: (shutil.rmtree(p), print('Deleting replay')), self.path)

        # Data specs

        self.frame_stack = frame_stack or 1
        obs_spec.shape[0] //= self.frame_stack

        self.specs = {'obs': obs_spec, 'action': action_spec, 'meta': {'shape': meta_shape},
                      **{name: {'shape': (1,)} for name in ['reward', 'label', 'step']}}

        # Episode traces (temporary in-RAM buffer until full episode ready to be stored)

        self.episode = {name: [] for name in self.specs}
        self.episode_len = 0
        self.episodes_stored = 0 if stream else len(list(self.path.glob('*.npz')))
        self.save = save

        self.offline = offline
        self.stream = stream  # Streaming from Environment directly

        # Data transform

        if transform is not None:
            if self.stream:
                warnings.warn('Command-line transforms (`transform=`) are not supported for streaming (`stream=true`). '
                              'Try a custom dataset instead via `Dataset=` defined with your preferred transforms.')
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
        self.nstep = 0 if suite == 'classify' or generate or stream else nstep

        if self.stream:
            return

        """
        ---Parallelized experience loading--- 
        
        Either Online or Offline. "Online" means the data size grows.
        
          Offline, data is adaptively pre-loaded onto CPU RAM from hard disk before training. Caching on RAM is faster 
          than loading from hard disk, epoch by epoch. We use truly-shared RAM memory. 

          The disadvantage of CPU pre-loading is the dependency on more CPU RAM. The "capacity=" a.k.a. :RAM_capacity=" 
          adapts how many experiences get stored on RAM vs. memory-mapped on hard disk. Memory mapping is an efficient 
          hard disk storage format for fast read-writes. 

          Online also caches data on RAM, after storing to hard disk. "capacity=" controls total capacity for training,
          forgetting the oldest data for training but still saving it on hard disk if "replay.save".
        """

        # CPU workers
        self.num_workers = max(1, min(num_workers, os.cpu_count()))

        os.environ['NUMEXPR_MAX_THREADS'] = str(self.num_workers)

        # RAM capacity per worker. Max num experiences allotted per CPU worker
        capacity = capacity // self.num_workers if capacity not in [-1, 'inf'] else np.inf

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

        self.batches = torch.utils.data.DataLoader(dataset=self.experiences,
                                                   batch_size=batch_size,
                                                   num_workers=self.num_workers,
                                                   pin_memory=True,
                                                   shuffle=offline,
                                                   worker_init_fn=worker_init_fn,
                                                   persistent_workers=True)

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
            except StopIteration:
                self.epoch += 1
                self._replay = None  # Reset iterator when depleted
                sample = next(self.replay)
            return *sample[:10 if trajectories else 6], *sample[10:]  # Return batch, w(/o) future-trajectories

    @property
    def replay(self):
        if self._replay is None:
            self._replay = iter(self.batches)  # Recreates the iterator when exhausted
        return self._replay

    # Initial iterator, allows replay iteration
    def __iter__(self):
        self._replay = iter(self.batches)
        return self.replay

    # Tracks single episode "trace" in memory buffer
    def add(self, experiences=None, store=False):
        if experiences is None:
            experiences = []

        # An "episode" or part of episode of experiences
        assert isinstance(experiences, (list, tuple))

        for exp in experiences:
            for name, spec in self.specs.items():
                # Missing data
                if name not in exp or exp[name] is None:
                    exp[name] = np.zeros((0,))

                # # Add batch dimension TODO If None, then should be 0-dim, not NaN
                if np.isscalar(exp[name]) or type(exp[name]) == bool:
                    exp[name] = np.full((1, *spec['shape']), exp[name], dtype=getattr(exp[name], 'dtype', 'float32'))
                # # elif len(exp[name].shape) in [0, 1, len(spec['shape'])]:
                # #     exp[name].shape = (1, *spec['shape'])  # Disabled for discrete/continuous conversions
                #
                # Expands attributes that are unique per batch (such as 'step')
                batch_size = exp.get('obs', exp['action']).shape[0]
                if 1 == exp[name].shape[0] < batch_size:
                    exp[name] = np.repeat(exp[name], batch_size, axis=0)

                # Validate consistency - disabled for discrete/continuous conversions
                # assert spec['shape'] == exp[name].shape[1:], \
                #     f'Unexpected shape for "{name}": {spec["shape"]} vs. {exp[name].shape[1:]}'
                spec['shape'] = exp[name].shape[1:]

                # Add the experience
                self.episode[name].append(exp[name])

            if self.stream:
                self.stream = exp  # For streaming directly from Environment TODO formatting... if above commented out

        # Count experiences in episode
        self.episode_len += len(experiences)

        if store:
            self.store_episode()  # Stores them in file system

    # Stores episode (to file in system)
    def store_episode(self):
        if self.episode_len == 0:
            return

        for name, spec in self.specs.items():
            # Concatenate into one big episode batch
            # Presumes a pre-existing batch dimension in each experience  # TODO SharedDict dtype
            self.episode[name] = np.concatenate(self.episode[name], axis=0).astype(np.float32)

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
        # Infinite if stream, else the number of episodes stored if constant or some deleted, else the episodes in path
        return int(5e11) if self.stream else self.episodes_stored if self.offline or not self.save \
            else len(list(self.path.glob('*.npz')))


# How to initialize each worker
def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


# A CPU worker that can iteratively and efficiently build/update batches of experience in parallel (from files/RAM)
class Experiences:
    def __init__(self, path, capacity, specs, fetch_per, pipes, save, offline, frame_stack, nstep, discount, transform):

        # Dataset construction via parallel workers

        self.path = path

        self.episode_names = []
        self.episodes = SharedDict(specs) if offline else dict()  # Episodes fetched on CPU

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

        # If Offline, share RAM across CPU workers
        if self.offline:
            self.num_experiences = sum([int(episode_name.stem.split('_')[-1]) - (self.nstep or 0)
                                        for episode_name in self.path.glob('*.npz')])

        self.initialized = False

    @property
    def worker_id(self):
        try:
            return torch.utils.data.get_worker_info().id
        except AttributeError:
            return 0

    @property
    def num_workers(self):
        return len(self.pipes)

    @property
    def pipe(self):
        return self.pipes[self.worker_id]

    def init_worker(self):
        self.worker_fetch_episodes()  # Load in existing data

        # If Offline, share RAM across CPU workers
        if self.offline:
            self.episode_names = sorted(self.path.glob('*.npz'))

            self.experience_indices = sum([list(enumerate([episode_name] * (int(episode_name.stem.split('_')[-1])
                                                                            - (self.nstep or 0))))
                                           for episode_name in self.episode_names], [])  # Slightly redundant per worker

        self.initialized = True

    def load_episode(self, episode_name):
        try:
            with episode_name.open('rb') as episode_file:
                episode = np.load(episode_file)
                episode = {key: episode[key] for key in episode.keys()}
        except:
            return False

        offset = self.nstep or 0
        episode_len = len(episode['obs']) - offset
        episode = {name: episode.get(name, np.zeros((0,))) for name, spec in self.specs.items()}

        episode['id'] = len(self.experience_indices)
        self.experience_indices += list(enumerate([episode_name] * episode_len))

        self.episode_names.append(episode_name)
        self.episode_names.sort()

        # If Offline replay exceeds RAM ("capacity=" flag) (approximately), keep episode on hard disk
        if self.offline and episode_len + len(self.experience_indices) > self.capacity:
            for spec in episode:
                if isinstance(episode[spec], np.ndarray) and episode[spec].size > 1:
                    path = self.path / f'{episode_name.stem}_{spec}.dat'
                    mmap_file = np.memmap(path, 'float32', 'w+', shape=episode[spec].shape)
                    mmap_file[:] = episode[spec][:]
                    mmap_file.flush()  # Write episode to hard disk
                    episode[spec] = mmap_file  # Replace episode with memory mapped link for efficient retrieval

            self.episodes[episode_name] = episode
            return True

        self.episodes[episode_name] = episode

        if not self.save:
            episode_name.unlink(missing_ok=True)  # Deletes file

        # Deleting experiences upon overfill if Online
        while episode_len + len(self) > self.capacity and not self.offline:
            early_episode_name = self.episode_names.pop(0)
            early_episode = self.episodes.pop(early_episode_name)
            early_episode_len = len(early_episode['obs']) - offset
            self.deleted_indices += early_episode_len  # To derive a consistent experience index even as data deleted

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
                        self.episodes[episode_name][key][idx] = update.numpy()  # Note: Only compatible with numpy data
                        # if isinstance(self.episodes[episode_name][key], np.memmmap):
                        #     self.episodes[episode_name][key].flush()  # Update in hard disk if memory mapped

    def sample(self, episode_names, metrics=None):
        episode_name = random.choice(episode_names)  # Uniform sampling of experiences
        return episode_name

    # N-step cumulative discounted rewards
    def process(self, episode, idx=None):
        offset = self.nstep or 0
        episode_len = len(episode['obs']) - offset
        if idx is None:
            idx = np.random.randint(episode_len)

        # Index into data that could be None
        def safe_index(data, idx=None, _from=None, _to=None):
            try:
                return data[_from:_to] if idx is None else data[idx]
            except IndexError:
                return data

        # Frame stack
        def frame_stack(traj_o, idx):
            frames = traj_o[max([0, idx + 1 - self.frame_stack]):idx + 1]
            for _ in range(self.frame_stack - idx - 1):  # If not enough frames, re-append first
                frames = np.concatenate([traj_o[:1], frames], 0)
            frames = frames.reshape(frames.shape[1] * self.frame_stack, *frames.shape[2:])
            return frames

        # Present
        obs = frame_stack(episode['obs'], idx)
        label = safe_index(episode['label'], idx)
        step = safe_index(episode['step'], idx)

        exp_id, worker_id = episode['id'] + idx, self.worker_id
        ids = np.array([exp_id, worker_id])

        meta = safe_index(episode['meta'], idx)  # Agent-writable Metadata

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
            traj_l = safe_index(episode['label'], _from=idx, _to=idx + self.nstep + 1)

            # Cumulative discounted reward
            discounts = self.discount ** np.arange(self.nstep + 1)
            reward = np.dot(discounts[:-1], traj_r)
            discount = discounts[-1:]
        else:
            action, reward = safe_index(episode['action'], idx), safe_index(episode['reward'], idx)

            next_obs = traj_o = traj_a = traj_r = traj_l = np.zeros(0,)
            discount = np.array([1.0])

        # Transform
        if self.transform is not None:
            obs = self.transform(torch.as_tensor(obs))

        return obs, action, reward, discount, next_obs, label, traj_o, traj_a, traj_r, traj_l, step, ids, meta

    def fetch_sample_process(self, idx=None):
        # Populate workers with up-to-date data
        if not self.initialized:
            self.init_worker()
        try:
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
        return self.num_experiences if self.offline else len(self.experience_indices) - self.deleted_indices


# Loads Experiences with an Iterable Dataset
class Online(Experiences, IterableDataset):
    def __init__(self, path, capacity, specs, fetch_per, pipes, save, frame_stack, nstep=0, discount=1, transform=None):
        super().__init__(path, capacity, specs, fetch_per, pipes, save, False, frame_stack, nstep, discount, transform)

    def __iter__(self):
        # Keep fetching, sampling, and building batches
        while True:
            yield self.fetch_sample_process()  # Yields a single experience


# Loads Experiences with a Map Style Dataset
class Offline(Experiences, Dataset):
    def __init__(self, path, capacity, specs, fetch_per, pipes, save, frame_stack, nstep=0, discount=1, transform=None):
        super().__init__(path, capacity, specs, fetch_per, pipes, save, True, frame_stack, nstep, discount, transform)

    def __getitem__(self, idx):
        # Retrieve a single experience by index
        return self.fetch_sample_process(idx)


class SharedDict:
    """
    An Offline dict of "episodes" dicts generalized to manage numpy arrays, integers, and hard disk memory map-links
    in truly-shared RAM memory efficiently read-writable across parallel CPU workers.
    """
    def __init__(self, specs):
        self.dict_id = str(uuid.uuid4())[:8]

        self.mems = {}
        self.specs = specs

        # Shared memory can create a lot of file descriptors
        _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

        # Increase soft limit to hard limit just in case
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard_limit, hard_limit))

    def __setitem__(self, key, value):
        self.start_worker()

        assert isinstance(value, dict), 'Shared Memory must be dict'

        num_episodes = key.stem.split('/')[-1].split('_')[1]

        for spec, data in value.items():
            name = self.dict_id + num_episodes + spec

            # if data is None:
            #     data = np.zeros((0,))

            # Shared integers
            if spec == 'id':
                mem = self.setdefault(name, [data])
                mem[0] = data
            # Shared numpy
            elif data.nbytes > 0:
                # Memory map link
                if isinstance(data, np.memmap):
                    self.mems[name] = data
                    self.setdefault(name + 'mmap', list(str(data.filename)))  # Constant per spec/episode
                # Shared RAM memory
                else:
                    mem = self.setdefault(name, data)
                    mem_ = np.ndarray(data.shape, dtype=data.dtype, buffer=mem.buf)
                    mem_[:] = data[:]
                    self.setdefault(name + 'mmap', [0])  # False, no memory mapping. Also expects constant

            # Data shape
            self.setdefault(name + 'shape', [1] if spec == 'id' else list(data.shape))  # Constant per spec/episode

    def __getitem__(self, key):
        # Account for potential delay
        for _ in range(2400):
            try:
                return self.get(key)
            except FileNotFoundError as e:
                sleep(1)
        raise(e)

    def get(self, key):
        num_episodes = key.stem.split('/')[-1].split('_')[1]

        episode = {}

        for spec in self.keys():
            name = self.dict_id + num_episodes + spec

            # Integer
            if spec == 'id':
                episode[spec] = int(self.getdefault(name, ShareableList)[0])
            # Numpy array
            else:
                # Shape
                shape = tuple(self.getdefault(name + 'shape', ShareableList))

                if 0 in shape:
                    episode[spec] = np.zeros((0,))  # Empty array
                else:
                    # Whether memory mapped
                    is_mmap = list(self.getdefault(name + 'mmap', ShareableList))

                    if 0 in is_mmap:
                        mem = self.getdefault(name, SharedMemory)
                        episode[spec] = np.ndarray(shape, np.float32, buffer=mem.buf)  # TODO dtype
                    else:
                        # Read from memory-mapped hard disk file rather than shared RAM
                        episode[spec] = self.getdefault(name, lambda **_: np.memmap(''.join(is_mmap), np.float32,
                                                                                    'r+', shape=shape))
        return episode

    def setdefault(self, name, data):
        if name not in self.mems:
            try:
                # Create shared memory link
                self.mems[name] = ShareableList(data, name=name) if isinstance(data, list) \
                    else SharedMemory(create=True, name=name,  size=data.nbytes)
            except FileExistsError:
                # Retrieve shared memory link
                self.mems[name] = (ShareableList if isinstance(data, list) else SharedMemory)(name=name)
        return self.mems[name]

    def getdefault(self, name, method):
        # Return if cached, else evaluate
        return self.mems[name] if name in self.mems else self.mems.setdefault(name, method(name=name))

    def keys(self):
        return self.specs.keys() | {'id'}

    def __del__(self):
        self.cleanup()

    def start_worker(self):
        # Hacky fix for https://bugs.python.org/issue38119
        if not self.mems:
            check_rtype = lambda func: lambda name, rtype: None if rtype == 'shared_memory' else func(name, rtype)
            resource_tracker.register = check_rtype(resource_tracker.register)
            resource_tracker.unregister = check_rtype(resource_tracker.unregister)

            if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
                del resource_tracker._CLEANUP_FUNCS["shared_memory"]

    def cleanup(self):
        for name, mem in self.mems.items():
            if isinstance(mem, ShareableList):
                mem = mem.shm
            if hasattr(mem, 'unlink'):
                mem.close()
                mem.unlink()
