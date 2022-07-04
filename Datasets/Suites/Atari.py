# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from collections import deque

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    import gym

import numpy as np

import torch

from torchvision.transforms.functional import resize

from skimage.transform import resize


# Access a dict with attribute or key (purely for aesthetic reasons)
class AttrDict(dict):
    def __init__(self, _dict):
        super(AttrDict, self).__init__()
        self.__dict__ = self
        self.update(_dict)


class Env:
    """
    A general-purpose environment.

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

    An "exp" (experience) is an AttrDict consisting of "obs", "action", "reward", "label", "step"
    numpy values which can be NaN. "obs" must include a batch dim.

    Can optionally include a frame_stack method.

    """
    def __init__(self, task='pong', seed=0, frame_stack=4,
                 screen_size=84, color='grayscale', sticky_action_proba=0, action_space_union=False,
                 last_2_frame_pool=True, terminal_on_life_loss=True, **kwargs):  # Atari-specific
        self.discrete = True
        self.episode_done = False

        # Make env

        task = f'ALE/{task}-v5'

        # Load task
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self.env = gym.make(task,
                                    obs_type=color,                   # ram | rgb | grayscale
                                    frameskip=1,                      # Frame skip  # Perhaps substitute action_repeat
                                    # mode=0,                         # Game mode, see Machado et al. 2018
                                    difficulty=0,                     # Game difficulty, see Machado et al. 2018
                                    repeat_action_probability=
                                    sticky_action_proba,              # Sticky action probability
                                    full_action_space=
                                    action_space_union,               # Use all actions
                                    render_mode=None                  # None | human | rgb_array
                                    )
        except gym.error.NameNotFound as e:
            # If Atari not installed
            raise gym.error.NameNotFound(str(e) + '\nYou may have not installed the Atari ROMs.\n'
                                                  'Try the following to install them, as in the README.\n'
                                                  'Accept the license:\n'
                                                  '$ pip install autorom\n'
                                                  '$ AutoROM --accept-license\n'
                                                  'Now, install ROMs:\n'
                                                  '$ mkdir ./Datasets/Suites/Atari_ROMS\n'
                                                  '$ AutoROM --install-dir ./Datasets/Suites/Atari_ROMS\n'
                                                  '$ ale-import-roms ./Datasets/Suites/Atari_ROMS\n'
                                                  'You should be good to go!')

        # Set random seed
        self.env.seed(seed)

        # Nature DQN-style pooling of last 2 frames
        self.last_2_frame_pool = last_2_frame_pool
        self.last_frame = None

        # Terminal on life loss
        self.terminal_on_life_loss = terminal_on_life_loss
        self.lives = None

        self.obs_spec = {'name': 'obs',
                         'shape': (frame_stack, screen_size, screen_size),
                         'mean': None,
                         'stddev': None,
                         'low': 0,
                         'high': 255}

        self.action_spec = {'name': 'action',
                            'shape': (1,),
                            'num_actions': self.env.action_space.n,
                            'low': None,
                            'high': None}

        self.frames = deque([], frame_stack or 1)

    def step(self, action):
        # Remove batch dim
        action = action.squeeze(0)

        # Step env
        obs, reward, self.episode_done, info = self.env.step(action)

        # Better than obs for some reason?  TODO testing; delete
        self.env.ale.getScreenGrayscale(obs)

        # Nature DQN-style pooling of last 2 frames
        if self.last_2_frame_pool:
            last_frame = self.last_frame
            self.last_frame = obs
            obs = np.maximum(obs, last_frame)

        # Terminal on life loss
        if self.terminal_on_life_loss:
            lives = self.env.ale.lives()
            if lives < self.lives:
                self.episode_done = True
            self.lives = lives

        obs = resize(obs, self.obs_spec['shape'][1:], preserve_range=True)
        obs = obs.astype(np.uint8)
        # Add channel dim
        obs = np.expand_dims(obs, axis=0)
        # Resize image
        # obs = resize(torch.as_tensor(obs), self.obs_spec['shape'][1:], antialias=True).numpy()
        # Add batch dim
        obs = np.expand_dims(obs, 0)

        # Create experience
        exp = {'obs': obs, 'action': action, 'reward': reward, 'label': None, 'step': None}

        # Scalars/NaN to numpy
        for key in exp:
            if np.isscalar(exp[key]) or exp[key] is None or type(exp[key]) == bool or exp[key].shape == ():
                exp[key] = np.full([1, 1], exp[key], 'float32')

        # Return experience
        return AttrDict(exp)

    def frame_stack(self, obs):
        for _ in range(self.frames.maxlen - len(self.frames) + 1):
            self.frames.append(obs)

        return np.concatenate(list(self.frames), axis=1)

    def reset(self):
        obs = self.env.reset()
        self.episode_done = False

        # Better than obs for some reason?  TODO testing; delete
        self.env.ale.getScreenGrayscale(obs)

        # Last frame
        if self.last_2_frame_pool:
            self.last_frame = obs

        # Lives
        if self.terminal_on_life_loss:
            self.lives = self.env.ale.lives()

        obs = resize(obs, self.obs_spec['shape'][1:], preserve_range=True)
        obs = obs.astype(np.uint8)
        # Add channel dim
        obs = np.expand_dims(obs, axis=0)
        # Resize image
        # obs = resize(torch.as_tensor(obs), self.obs_spec['shape'][1:], antialias=True).numpy()
        # Add batch dim
        obs = np.expand_dims(obs, 0)

        # Create experience
        exp = {'obs': obs, 'action': None, 'reward': 0, 'label': None, 'step': None}

        # Scalars/NaN to numpy
        for key in exp:
            if np.isscalar(exp[key]) or exp[key] is None or type(exp[key]) == bool or exp[key].shape == ():
                exp[key] = np.full([1, 1], exp[key], 'float32')

        # Clear frame stack
        self.frames.clear()

        # Return experience
        return AttrDict(exp)

    def render(self):
        return self.env.render('rgb_array')  # rgb_array | human
