# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
from math import inf

from Datasets.Suites import DMC, Atari, Classify


class Environment:
    def __init__(self, task_name, frame_stack, action_repeat, max_episode_steps=inf, truncate_episode_steps=inf,
                 suite="DMC", offline=False, generate=False, train=True, seed=0, **kwargs):
        self.suite = suite
        self.offline = offline
        self.generate = generate

        # Offline and generate don't use training rollouts
        self.disable = (offline or generate) and train

        if not self.disable:
            self.action_repeat = action_repeat or 1

            self.max_episode_steps = train and max_episode_steps or inf  # inf default
            self.truncate_episode_steps = train and truncate_episode_steps or inf  # inf default

            self.env = self.raw_env.Env(task_name, seed, frame_stack, action_repeat, **kwargs)  # TODO Classify make train set with eval

            self.exp = self.env.reset()

        self.episode_done = self.episode_step = self.episode_frame = self.last_episode_len = self.episode_reward = 0
        self.daybreak = None

    @property
    def raw_env(self):
        if self.suite.lower() == "dmc":
            return DMC
        elif self.suite.lower() == "atari":
            return Atari
        elif self.suite.lower() == 'classify':
            return Classify

    def __getattr__(self, item):
        return getattr(self.env, item)

    def rollout(self, agent, steps=inf, vlog=False):
        if self.daybreak is None:
            self.daybreak = time.time()  # "Daybreak" for whole episode

        experiences = []
        video_image = []

        self.episode_done = self.disable

        step = frame = 0
        while not self.episode_done and step < steps:
            obs = getattr(self.env, 'frame_stack',
                          self.exp.obs)

            # Act
            action = agent.act(obs)

            if not self.generate:
                self.exp = self.env.step(action.cpu().numpy())

            self.exp.step = agent.step
            experiences.append(self.exp)

            if vlog or self.generate:
                image_frame = action[:24].view(-1, *obs.shape[1:]) if self.generate \
                    else self.env.render()
                video_image.append(image_frame)

            step += 1
            frame += len(action)

            # Tally reward
            self.episode_reward += self.exp.reward.mean()

            self.episode_done = \
                self.env.episode_done or self.episode_step > min(self.max_episode_steps,
                                                                 self.truncate_episode_steps) - 2 or self.generate

            if self.env.episode_done or self.episode_step > self.max_episode_steps - 2:
                self.exp = self.env.reset()

        # Tally time
        self.episode_step += step
        self.episode_frame += frame

        agent.episode += self.episode_done * agent.training

        # TODO for now
        if self.episode_done:
            self.last_episode_len = self.episode_step

        # Log stats
        sundown = time.time()
        frames = self.episode_frame * self.action_repeat

        logs = {'time': sundown - agent.birthday,
                'step': agent.step,
                'frame': agent.frame * self.action_repeat,
                'epoch' if self.offline or self.generate else 'episode': agent.epoch if self.offline or self.generate else agent.episode,
                'accuracy'if self.suite == 'classify' else 'reward':
                    self.episode_reward / max(1, self.episode_step * self.suite == 'classify'),  # Accuracy is %
                'fps': frames / (sundown - self.daybreak)} if not self.disable \
            else None

        if self.episode_done:
            self.episode_step = self.episode_frame = self.episode_reward = 0
            self.daybreak = sundown

        return experiences, logs, video_image
