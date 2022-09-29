# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from collections import deque

import warnings

# Super Mario environment for OpenAI Gym
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    import gym_super_mario_bros

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

import numpy as np

from torch import as_tensor

from torchvision.transforms.functional import resize, rgb_to_grayscale


class SuperMario:
    """
    A general-purpose environment:

    Must accept: **kwargs as init arg.

    Must have:

    (1) a "step" function, action -> exp
    (2) "reset" function, -> exp
    (3) "render" function, -> image
    (4) "episode_done" attribute
    (5) "obs_spec" attribute which includes:
        - "shape", "mean", "stddev", "low", "high" (the last 4 can be None)
    (6) "action-spec" attribute which includes:
        - "shape", "discrete_bins" (should be None if not discrete), "low", "high", and "discrete"
    (7) "exp" attribute containing the latest exp

    Recommended: Discrete environments should have a conversion strategy for adapting continuous actions (e.g. argmax)

    An "exp" (experience) is an AttrDict consisting of "obs", "action" (prior to adapting), "reward", "label", "step"
    numpy values which can be NaN. Must include a batch dim.

    ---

    Can optionally include a frame_stack, action_repeat method.

    """
    def __init__(self, world=1, stage=1, version='v0', seed=0, frame_stack=4, action_repeat=4,
                 screen_size=84, color='grayscale', last_2_frame_pool=False, **kwargs):
        self.episode_done = False

        # Make env

        # task = SuperMarioBros-[World]-[Stage]-[Version]: SuperMarioBros-[1 - 8]-[1 - 4]-[0 - 3]
        task = f'SuperMarioBros-{world}-{stage}-{version}'

        # Load task
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            env = gym_super_mario_bros.make(task)

            # Limit the action-space to (0) Walk right and (1) Jump right
            self.env = JoypadSpace(env, [["right"], ["right", "A"]])

        # Set random seed
        self.env.seed(seed)

        # Nature DQN-style pooling of last 2 frames
        self.last_2_frame_pool = last_2_frame_pool
        self.last_frame = None

        # Number of channels
        self.color = color
        channels = 3 if color == 'rgb' else 1

        self.obs_spec = {'shape': (channels * frame_stack, screen_size, screen_size),
                         'mean': None,
                         'stddev': None,
                         'low': 0,
                         'high': 255}

        self.action_spec = {'shape': (1,),
                            'discrete_bins': self.env.action_space.n,
                            'low': 0,
                            'high': self.env.action_space.n - 1,
                            'discrete': True}

        self.exp = None

        self.action_repeat = action_repeat or 1
        self.frames = deque([], frame_stack or 1)

    def step(self, action):
        # Adapt to discrete!
        _action = self.adapt_to_discrete(action)
        _action.shape = self.action_spec['shape']

        # Step env
        reward = 0
        for _ in range(self.action_repeat):
            obs, _reward, self.episode_done, info = self.env.step(int(_action))  # SuperMario requires scalar int action
            reward += _reward
            if self.last_2_frame_pool:
                last_frame = self.last_frame
                self.last_frame = obs
            if self.episode_done:
                break

        # Nature DQN-style pooling of last 2 frames
        if self.last_2_frame_pool:
            obs = np.maximum(obs, last_frame)

        # Image channels
        obs = as_tensor(obs.transpose(2, 0, 1).copy())  # Channel-first

        if self.color == 'grayscale':
            obs = rgb_to_grayscale(obs)

        # Resize image
        obs = resize(obs, self.obs_spec['shape'][1:], antialias=True).numpy()

        # Add batch dim
        obs = np.expand_dims(obs, 0)

        # Create experience
        exp = {'obs': obs, 'action': action, 'reward': reward, 'label': None, 'step': None}

        # Scalars/NaN to numpy
        for key in exp:
            if np.isscalar(exp[key]) or exp[key] is None or type(exp[key]) == bool:
                exp[key] = np.full([1, 1], exp[key], dtype=getattr(exp[key], 'dtype', 'float32'))
            elif len(exp[key].shape) in [0, 1]:  # Add batch dim
                exp[key].shape = (1, *(exp[key].shape or [1]))

        self.exp = AttrDict(exp)  # Experience

        return self.exp

    def frame_stack(self, obs):
        if self.frames.maxlen == 1:
            return obs

        self.frames.extend([obs] * (self.frames.maxlen - len(self.frames) + 1))
        return np.concatenate(list(self.frames), axis=1)

    def reset(self):
        obs = self.env.reset()
        self.episode_done = False

        # Last frame
        if self.last_2_frame_pool:
            self.last_frame = obs

        # Image channels
        obs = as_tensor(obs.transpose(2, 0, 1).copy())  # Channel-first

        if self.color == 'grayscale':
            obs = rgb_to_grayscale(obs)

        # Resize image
        obs = resize(obs, self.obs_spec['shape'][1:], antialias=True).numpy()

        # Add batch dim
        obs = np.expand_dims(obs, 0)

        # Create experience
        exp = {'obs': obs, 'action': None, 'reward': 0, 'label': None, 'step': None}

        # Scalars/NaN to numpy
        for key in exp:
            if np.isscalar(exp[key]) or exp[key] is None or type(exp[key]) == bool:
                exp[key] = np.full([1, 1], exp[key], dtype=getattr(exp[key], 'dtype', 'float32'))
            elif len(exp[key].shape) in [0, 1]:  # Add batch dim
                exp[key].shape = (1, *(exp[key].shape or [1]))

        # Reset frame stack
        self.frames.clear()

        self.exp = AttrDict(exp)  # Experience

        return self.exp

    def render(self):
        return self.env.render()  # rgb_array | human

    def adapt_to_discrete(self, action):
        shape = self.action_spec['shape']

        try:
            action = action.reshape(len(action), *shape)  # Assumes a batch dim
        except (ValueError, RuntimeError):
            try:
                action = action.reshape(len(action), -1, *shape)  # Assumes a batch dim
            except:
                raise RuntimeError(f'Discrete environment could not broadcast or adapt action of shape {action.shape} '
                                   f'to expected batch-action shape {(-1, *shape)}')
            action = action.argmax(1)

        discrete_bins, low, high = self.action_spec['discrete_bins'], self.action_spec['low'], self.action_spec['high']

        # Round to nearest decimal/int corresponding to discrete bins, high, and low
        return np.round((action - low) / (high - low) * (discrete_bins - 1)) / (discrete_bins - 1) * (high - low) + low


# Access a dict with attribute or key (purely for aesthetic reasons)
class AttrDict(dict):
    def __init__(self, _dict):
        super(AttrDict, self).__init__()
        self.__dict__ = self
        self.update(_dict)
