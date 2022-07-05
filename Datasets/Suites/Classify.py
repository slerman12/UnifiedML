# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import datetime
import glob
import io
import itertools
import json
import random
import warnings
from pathlib import Path
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision.transforms import functional as F

from Utils import instantiate


# Access a dict with attribute or key (purely for aesthetic reasons)
class AttrDict(dict):
    def __init__(self, _dict):
        super(AttrDict, self).__init__()
        self.__dict__ = self
        self.update(_dict)


class Classify:
    """
    A general-purpose environment:

    Must accept: (task, seed, **kwargs) as init args.

    Must have:

    (1) a "step" function, action -> exp
    (2) "reset" function, -> exp
    (3) "render" function, -> image
    (4) "discrete" attribute
    (5) "episode_done" attribute
    (6) "obs_spec" attribute which includes:
        - "name" ('obs'), "shape", "mean", "stddev", "low", "high" (the last 4 can be None)
    (7) "action-spec" attribute which includes:
        - "name" ('action'), "shape", "num_actions" (should be None if not discrete),
          "low", "high" (these last 2 should be None if discrete, can be None if not discrete)
    (8) "exp" attribute containing the latest exp

    An "exp" (experience) is an AttrDict consisting of "obs", "action", "reward", "label", "step"
    numpy values which can be NaN. "obs" must include a batch dim.

    ---

    Extended to accept a dataset config, which instantiates a Dataset. Datasets must:
    - extend Pytorch Datasets
    - include a "classes" (num classes) attribute
    - output (obs, label) pairs

    An "evaluate_episodes" attribute divides evaluation across batches

    """
    def __init__(self, dataset, task='MNIST', train=True, offline=True, generate=False, batch_size=32, num_workers=1,
                 low=None, high=None, frame_stack=None, action_repeat=None, seed=None, **kwargs):
        self.discrete = False
        self.episode_done = False

        # Make env

        # with warnings.catch_warnings():
        #     warnings.filterwarnings('ignore', '.*The given NumPy array.*')

        # Different datasets have different specs
        root_specs = [dict(root=f'./Datasets/ReplayBuffer/Classify/{task}_%s' %
                                ('Train' if train else 'Eval')), {}]
        train_specs = [dict(version='2021_' + 'train' if train else 'valid'),
                       dict(train=train), {}]
        download_specs = [dict(download=True), {}]
        transform_specs = [dict(transform=Transform()), {}]

        # Instantiate dataset
        for all_specs in itertools.product(root_specs, train_specs, download_specs, transform_specs):
            try:
                root_spec, train_spec, download_spec, transform_spec = all_specs
                specs = dict(**root_spec, **train_spec, **download_spec, **transform_spec)
                specs.update(kwargs)
                dataset = instantiate(dataset, **specs)
            except TypeError:
                continue
            break

        assert isinstance(dataset, Dataset), 'Dataset must be a Pytorch Dataset or inherit from a Pytorch Dataset'

        self.action_spec = {'name': 'action',
                            'shape': (len(dataset.classes),),  # Dataset must include a "classes" attr
                            'num_actions': None,
                            'low': None,
                            'high': None}

        self.batches = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=train,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn)

        self._batches = iter(self.batches)

        # Offline and generate don't do training rollouts
        if (offline or generate) and not train:
            # Call training version
            Classify(dataset, task, True, offline, generate, batch_size, num_workers, None, None, None, None, None,
                     **kwargs)

        # Create replay
        if train and (offline or generate):
            replay_path = Path(f'./Datasets/ReplayBuffer/Classify/{task}_Buffer')
            if offline and not replay_path.exists():
                self.create_replay(replay_path)

        # Compute stats
        stats_path = glob.glob(f'./Datasets/ReplayBuffer/Classify/{task}_Stats_*')
        mean, stddev, low_, high_ = map(json.loads, stats_path[0].split('_')[-4:]) if len(stats_path) \
            else self.compute_stats(f'./Datasets/ReplayBuffer/Classify/{task}') if train \
            else (None, None, low, high)
        low, high = low if low is None else low_, high_ if high is None else high

        self.obs_spec = {'name': 'obs',
                         'shape': tuple(next(iter(self.batches))[0].shape[1:]),
                         'mean': mean,
                         'stddev': stddev,
                         'low': low,
                         'high': high}

        self.exp = None  # Experience

        self.evaluate_episodes = len(self.batches)

        # Offline and generate don't do training rollouts, no need to waste memory
        if (offline or generate) and train:
            self.batches = self._batches = dataset = None

    def step(self, action):
        correct = (self.exp.label == np.expand_dims(np.argmax(action, -1), 1)).astype('float32')

        self.exp.reward = correct
        self.exp.action = action

        self.episode_done = True

        return self.exp

    def reset(self):
        obs, label = [np.array(b, dtype='float32') for b in self.sample()]
        label = np.expand_dims(label, 1)

        self.episode_done = False

        # Create experience
        exp = {'obs': obs, 'action': None, 'reward': None, 'label': label, 'step': None}

        self.exp = AttrDict(exp)  # Experience

        return self.exp

    def render(self):
        # Assumes image dataset
        image = self.sample()[0] if self.exp is None else self.exp.obs
        return np.array(image[random.randint(0, len(image))], dtype='uint8').transpose(1, 2, 0)

    def sample(self):
        try:
            return next(self._batches)
        except StopIteration:
            self._batches = iter(self.batches)
            return next(self._batches)

    def create_replay(self, path):
        path.mkdir(exist_ok=True, parents=True)

        for episode_ind, (obs, label) in enumerate(tqdm(self.batches, 'Creating a universal replay for this dataset. '
                                                                      'This only has to be done once')):
            obs, label = [np.array(b, dtype='float32') for b in (obs, label)]
            label = np.expand_dims(label, 1)

            episode = {'obs': obs, 'action': None, 'reward': None, 'label': label, 'step': None}

            timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            episode_name = f'{timestamp}_{episode_ind}_{obs.shape[0]}.npz'

            with io.BytesIO() as buffer:
                np.savez_compressed(buffer, **episode)
                buffer.seek(0)
                with (path / episode_name).open('wb') as f:
                    f.write(buffer.read())

    def compute_stats(self, path):
        cnt = 0
        fst_moment, snd_moment = None, None
        low, high = np.inf, -np.inf

        for obs, _ in tqdm(self.batches, 'Computing mean, stddev, min, max for standardization/normalization. '
                                         'This only has to be done once'):
            b, c, h, w = obs.shape
            fst_moment = torch.empty(c) if fst_moment is None else fst_moment
            snd_moment = torch.empty(c) if snd_moment is None else snd_moment
            nb_pixels = b * h * w
            sum_ = torch.sum(obs, dim=[0, 2, 3])
            sum_of_square = torch.sum(obs ** 2, dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

            cnt += nb_pixels

            low, high = min(obs.min(), low), max(obs.max(), high)

        mean, stddev = fst_moment.tolist(), torch.sqrt(snd_moment - fst_moment ** 2).tolist()
        open(path + f'_Stats_{mean}_{stddev}_{low}_{high}', 'w')  # Save stat values for future reuse

        return mean, stddev, low.item(), high.item()


class Transform:
    def __call__(self, sample):
        return F.to_tensor(sample)


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)

