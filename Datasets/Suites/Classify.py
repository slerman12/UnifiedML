# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import warnings
from pathlib import Path

from tqdm import tqdm
import numpy as np

from dm_env import specs, StepType

import torch
import torchvision
from torchvision.transforms import functional as F

from Datasets.Suites._Wrappers import ActionSpecWrapper, AugmentAttributesWrapper, ExtendedTimeStep

from Datasets.ReplayBuffer.Classify._TinyImageNet import TinyImageNet


class ClassifyEnv:
    def __init__(self, experiences, batch_size, num_workers, offline, train, buffer_path=None):

        def worker_init_fn(worker_id):
            seed = np.random.get_state()[1][0] + worker_id
            np.random.seed(seed)
            random.seed(seed)

        self.num_classes = len(experiences.classes)
        self.action_repeat = 1

        self.batches = torch.utils.data.DataLoader(dataset=experiences,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=num_workers,
                                                   pin_memory=True,
                                                   worker_init_fn=worker_init_fn)

        self._batches = iter(self.batches)

        if train:
            if offline and not buffer_path.exists():
                self.create_replay(buffer_path)
        else:
            self.evaluate_episodes = len(self)

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
                                                                'This only has to be done once. '
                                                                'Loading in batches, this may take a moment')):
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

    # ExperienceReplay expects at least a reset state and 'next obs', with 'reward' paired with (<->) 'next obs'
    def step(self, action):
        assert self.time_step.observation.shape[0] == action.shape[0], 'Agent must produce actions for each obs'

        # Concat a dummy batch item ('next obs')
        x, y = [np.concatenate([b, b[:1]], 0) for b in (self.time_step.observation, self.time_step.label)]

        correct = (self.time_step.label == np.expand_dims(np.argmax(action, -1), 1)).astype('float32')

        # 'reward' and 'action' paired with 'next obs'
        self.time_step.reward[1:] = correct
        self.time_step.reward[0] = correct.mean()
        self.time_step.action[1:] = action

        self.time_step = self.time_step._replace(step_type=StepType.LAST, observation=x, label=y)

        return self.time_step

    def observation_spec(self):
        if not hasattr(self, 'observation'):
            self.observation = np.array(self.batch[0])
        return specs.BoundedArray(self.observation.shape[1:], self.observation.dtype, 0, 255, 'observation')

    def action_spec(self):
        return specs.BoundedArray((self.num_classes,), 'float32', 0, self.num_classes - 1, 'action')

    def __len__(self):
        return len(self.batches)


def make(task, frame_stack=4, action_repeat=4, episode_max_frames=False, episode_truncate_resume_frames=False,
         offline=False, train=True, seed=1, batch_size=1, num_workers=1):
    """
    'task' options:

    ('LSUN', 'LSUNClass',
     'ImageFolder', 'DatasetFolder', 'FakeData',
     'CocoCaptions', 'CocoDetection',
     'CIFAR10', 'CIFAR100', 'EMNIST', 'FashionMNIST', 'QMNIST',
     'MNIST', 'KMNIST', 'STL10', 'SVHN', 'PhotoTour', 'SEMEION',
     'Omniglot', 'SBU', 'Flickr8k', 'Flickr30k',
     'VOCSegmentation', 'VOCDetection', 'Cityscapes', 'ImageNet',
     'Caltech101', 'Caltech256', 'CelebA', 'WIDERFace', 'SBDataset',
     'VisionDataset', 'USPS', 'Kinetics400', 'HMDB51', 'UCF101',
     'Places365', 'TinyImageNet')
    """

    assert task in torchvision.datasets.__all__ or task == 'TinyImageNet'

    dataset = TinyImageNet if task == 'TinyImageNet' else getattr(torchvision.datasets, task)

    path = f'./Datasets/ReplayBuffer/Classify/{task}'

    class Transform:
        def __call__(self, sample):
            sample = F.to_tensor(sample)
            sample *= 255  # Encoder expects pixels
            return sample

    transform = Transform()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*The given NumPy array.*')

        experiences = dataset(root=path + "_Train" if train else "_Eval",
                              train=train,
                              download=True,
                              transform=transform)

    create_replay_path = Path(path + '_Buffer')

    env = ClassifyEnv(experiences, batch_size, num_workers, offline, train, create_replay_path)

    env = ActionSpecWrapper(env, env.action_spec().dtype, discrete=False)
    env = AugmentAttributesWrapper(env,
                                   add_remove_batch_dim=False)  # Disables the modification of batch dims

    return env
