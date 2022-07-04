# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
from math import inf

from Datasets.Suites import DMC, Atari, Classify


class Environment:
    def __init__(self, task_name, frame_stack, action_repeat, max_episode_frames=inf, truncate_episode_frames=inf,
                 suite="DMC", offline=False, generate=False, train=True, seed=0, **kwargs):
        self.suite = suite
        self.offline = offline
        self.disable = (offline or generate) and train
        self.generate = generate

        self.action_repeat = action_repeat or 1

        self.max_episode_steps = train and max_episode_frames and max_episode_frames // action_repeat or inf
        self.truncate_episode_steps = train and truncate_episode_frames and truncate_episode_frames // action_repeat or inf

        self.env = self.raw_env.Env(task_name, seed, frame_stack, action_repeat, **kwargs)

        if not self.disable:
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
            obs = getattr(self.env, 'frame_stack', self.exp.obs)

            # Act
            action = agent.act(obs)

            self.exp = self.env.step(action.cpu().numpy()) if not self.generate else self.exp

            self.exp.step = agent.step
            experiences.append(self.exp)

            if vlog or self.generate:
                image_frame = action[:24].view(-1, *obs.shape[1:]) if self.generate \
                    else self.env.render()
                video_image.append(image_frame)

            step += 1
            frame += len(action)

            # Tally reward, done
            self.episode_reward += self.exp.reward.mean()
            self.episode_done = self.env.episode_done or self.generate

            self.episode_done = self.episode_done or self.episode_step + step == self.max_episode_steps
            if self.episode_step + step == self.truncate_episode_steps:
                break

        self.episode_step += step
        self.episode_frame += frame

        if self.episode_done or self.episode_step == self.truncate_episode_steps:
            if agent.training:
                agent.episode += 1

            if self.episode_done and not self.disable:
                self.exp = self.env.reset()
            self.last_episode_len = self.episode_frame

        self.episode_done = self.episode_done or self.episode_step == self.truncate_episode_steps

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
