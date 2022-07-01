# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import datetime
import glob
import io
import json
import random
import warnings
from pathlib import Path
from tqdm import tqdm

from hydra.utils import instantiate

from dm_env import specs, StepType

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision.transforms import functional as F

from Datasets.Suites._Wrappers import ActionSpecWrapper, AugmentAttributesWrapper, ExtendedTimeStep
from Datasets.ReplayBuffer.Classify._TinyImageNet import TinyImageNet


class ClassifyEnv:
    """A classification environment"""
    def __init__(self, experiences, batch_size, num_workers, offline, train, path=None):

        self.num_classes = len(experiences.classes)
        self.action_repeat = 1

        if not train:
            # Give eval equal-sized batches for easy accuracy computation
            batch_size = max([i for i in range(1, batch_size + 1) if len(experiences) % i == 0][-1], batch_size // 2)

        self.batches = DataLoader(dataset=experiences,
                                  batch_size=batch_size,
                                  shuffle=train,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn)

        self._batches = iter(self.batches)

        buffer_path = Path(path + '_Buffer')

        if train:
            if offline and not buffer_path.exists():
                self.create_replay(buffer_path)
        else:
            self.evaluate_episodes = len(self)

        norm_path = glob.glob(path + '_Normalization_*')

        if len(norm_path):
            mean, stddev = map(json.loads, norm_path[0].split('_')[-2:])
            self.data_stats = [mean, stddev]
        elif train:
            self.compute_norm(path)
        else:
            assert False

        self.min, self.max = [0] * self.observation_spec().shape[0], [1] * self.observation_spec().shape[0]
        self.data_stats += [self.min, self.max]

        # TODO do batches all make it into first epoch, reset iter?
        # self._batches = iter(self.batches)

        # No need to waste memory
        if offline and train:
            self.batches = self._batches = None

    @property
    def batch(self):
        try:
            batch = next(self._batches)
        except StopIteration:
            self._batches = iter(self.batches)
            batch = next(self._batches)
        return batch

    def create_replay(self, path):
        path.mkdir(exist_ok=True, parents=True)

        for episode_ind, (x, y) in enumerate(tqdm(self.batches, 'Creating a universal replay for this dataset. '
                                                                'This only has to be done once')):
            x, y, dummy_action, dummy_reward, dummy_discount, dummy_step = self.reset_format(x, y)

            # Concat a dummy batch item
            x, y = [np.concatenate([b, np.full_like(b[:1], np.NaN)], 0) for b in (x, y)]

            episode = {'observation': x, 'action': dummy_action, 'reward': dummy_reward, 'discount': dummy_discount,
                       'label': y, 'step': dummy_step}

            timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            episode_name = f'{timestamp}_{episode_ind}_{len(x)}.npz'

            with io.BytesIO() as buffer:
                np.savez_compressed(buffer, **episode)
                buffer.seek(0)
                with (path / episode_name).open('wb') as f:
                    f.write(buffer.read())

    def compute_norm(self, path):
        cnt = 0
        fst_moment, snd_moment = None, None

        for x, _ in tqdm(self.batches, 'Computing mean and stddev for normalization. '
                                       'This only has to be done once'):
            b, c, h, w = x.shape
            fst_moment = torch.empty(c) if fst_moment is None else fst_moment
            snd_moment = torch.empty(c) if snd_moment is None else snd_moment
            nb_pixels = b * h * w
            sum_ = torch.sum(x, dim=[0, 2, 3])
            sum_of_square = torch.sum(x ** 2, dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

            cnt += nb_pixels

        self.data_stats = [fst_moment.tolist(), torch.sqrt(snd_moment - fst_moment ** 2).tolist()]
        open(path + f'_Normalization_{self.data_stats[0]}_{self.data_stats[1]}', 'w')  # Save norm values for future reuse TODO standardization

    def reset_format(self, x, y):
        x, y = [np.array(b, dtype='float32') for b in (x, y)]
        y = np.expand_dims(y, 1)

        batch_size = x.shape[0]

        dummy_action = np.full([batch_size + 1, self.num_classes], np.NaN, 'float32')
        dummy_reward = dummy_step = np.full([batch_size + 1, 1], np.NaN, 'float32')
        dummy_discount = np.full([batch_size + 1, 1], 1, 'float32')

        return x, y, dummy_action, dummy_reward, dummy_discount, dummy_step

    def reset(self):
        x, y, dummy_action, dummy_reward, dummy_discount, dummy_step = self.reset_format(*self.batch)

        self.time_step = ExtendedTimeStep(reward=dummy_reward, action=dummy_action,
                                          discount=dummy_discount, step=dummy_step,
                                          step_type=StepType.FIRST, observation=x, label=y)

        return self.time_step

    # ExperienceReplay expects at least a reset state and 'next obs', with 'reward' index-paired with (<->) 'next obs'
    def step(self, action=None):
        if action is not None:
            assert self.time_step.observation.shape[0] == action.shape[0], 'Agent must provide actions for obs'

        # Concat a dummy batch item ('next obs') TODO why is first eval batch 0?
        x, y = [np.concatenate([b, b[:1]], 0) for b in (self.time_step.observation, self.time_step.label)]

        correct = np.full_like(self.time_step.label, np.NaN) if action is None \
            else (self.time_step.label == np.expand_dims(np.argmax(action, -1), 1)).astype('float32')

        # 'reward' and 'action' paired with 'next obs'
        self.time_step.reward[1:] = correct
        self.time_step.reward[0] = correct.mean()
        self.time_step.action[1:] = self.time_step.action[1:] if action is None else action

        self.time_step = self.time_step._replace(step_type=StepType.LAST, observation=x, label=y)

        return self.time_step

    def render(self):
        image = self.time_step.x if hasattr(self.time_step, 'x') \
            else self.batch[0]
        return np.array(image[random.randint(0, len(image))], dtype='uint8').transpose(1, 2, 0)

    def observation_spec(self):
        if not hasattr(self, 'observation'):
            self.observation = np.array(self.batch[0])
        return specs.BoundedArray(self.observation.shape[1:], self.observation.dtype, 0, 255, 'observation')

    def action_spec(self):
        return specs.BoundedArray((self.num_classes,), 'float32', 0, self.num_classes - 1, 'action')

    def __len__(self):
        return len(self.batches)


def make(task, dataset, frame_stack=4, action_repeat=4, episode_max_frames=False, episode_truncate_resume_frames=False,
         offline=False, train=True, seed=1, batch_size=1, num_workers=1):
    """
    'task' options:

    ('CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
    'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
    'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
    'VOCSegmentation', 'VOCDetection', 'Cityscapes', 'ImageNet',
    'Caltech101', 'Caltech256', 'CelebA', 'WIDERFace', 'SBDataset',
    'USPS', 'Kinetics400', "Kinetics", 'HMDB51', 'UCF101',
    'Places365', 'Kitti', "INaturalist", "LFWPeople", "LFWPairs",
    'TinyImageNet')
    """

    assert task in torchvision.datasets.__all__ or task == 'TinyImageNet' or 'Custom.' in task

    # TODO clean
    if 'Custom.' not in task:
        dataset_class = TinyImageNet if task == 'TinyImageNet' else getattr(torchvision.datasets, task)

    path = f'./Datasets/ReplayBuffer/Classify/{task}'

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*The given NumPy array.*')

        assert dataset._target_ or 'Custom.' not in task, 'Custom task must specify the `Dataset=` flag'

        if dataset._target_ and 'Custom.' not in task and train:
            print(f'Setting train dataset to {dataset._target_}.\n'
                  f'Note: to also set eval, set `task=classify/custom`. Eval: {task}')

        # If custom, should override environment.dataset and generalize.dataset, otherwise just environment.dataset
        experiences = instantiate(dataset, train=train, transform=Transform()) if dataset._target_ and \
                                                                                  ('Custom.' in task or train) \
            else dataset_class(root=path + "_Train" if train else path + "_Eval",
                               **(dict(version=f'2021_{"train" if train else "valid"}') if task == 'INaturalist'
                                  else dict(train=train)),
                               download=True,
                               transform=Transform())

        assert isinstance(experiences, Dataset), 'Dataset must be a Pytorch Dataset or inherit from a Pytorch Dataset'
        assert hasattr(experiences, 'classes'), 'Classify Dataset must define a "classes" attribute'

    env = ClassifyEnv(experiences, batch_size, num_workers, offline, train, path)

    env = ActionSpecWrapper(env, env.action_spec().dtype, discrete=False)
    env = AugmentAttributesWrapper(env,
                                   add_remove_batch_dim=False)  # Disables the modification of batch dims

    return env


class Transform:
    def __call__(self, sample):
        # Convert 1d to 2d  TODO not for proprioceptive? move to encoder?
        if hasattr(sample, 'shape'):
            while len(sample.shape) < 3:
                sample = np.expand_dims(sample, -1)  # Add spatial dims
        sample = F.to_tensor(sample)
        return sample


def worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)
