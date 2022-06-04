# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
from math import inf

from hydra.utils import instantiate


class World:
    def __init__(self, suite, environments, num_environments=1, offline=False, generate=False, train=True, seed=0):
        self.suite = suite

        self.num_environments = num_environments

        self.disable = (offline or generate) and train
        self.generate = generate

        # self.action_repeat = action_repeat  # Repeat the same action times
        # self.episode_max_frames = episode_max_frames  # When to reset an episode early
        # self.episode_truncate_resume_frames = episode_truncate_resume_frames  # When toggle episode_done then resume

        # self.world = World(suite, task_name, frame_stack, seed, train, offline, num_environments, batch_size,
        #                    num_workers, action_repeat=1, episode_max_frames=False, episode_truncate_resume_frames=800)
        self.envs = instantiate(environments, train=train, seed=seed)

        self.exps = self.envs.init()  # Experiences

        self.episode_done = self.episode_step = self.last_episode_len = self.episode_reward = 0
        self.daybreak = None

    def __getattr__(self, item):
        return getattr(self.world, item)

    def rollout(self, agent, steps=inf, vlog=False):
        if self.daybreak is None:
            self.daybreak = time.time()  # "Daybreak" for whole episode

        experiences = []
        video_image = []

        self.episode_done = [self.disable] * self.num_environments

        step = 0
        while not self.episode_done and step < steps:
            # Act
            action = agent.act(self.exps.observation)

            exps = self.exps = self.envs.step(None if self.generate else action.cpu().numpy())

            exps.step = agent.step
            experiences.append((exps.observation, exps.action, exps.reward, exps.label))

            if vlog or self.generate:
                frame = action[:24].view(-1, *exps.observation.shape[1:]) if self.generate \
                    else self.env.physics.render(height=256, width=256, camera_id=0) \
                    if hasattr(self.env, 'physics') else self.env.render()
                video_image.append(frame)

            # Tally reward, done
            self.episode_reward += exps.reward.mean()
            self.episode_done = exps.episode_done

            step += 1

        self.episode_step += step  # TODO * batch_size?

        if self.disable:
            agent.step += 1  # TODO agent should iterate by batch size

        if self.episode_done and not self.disable:
            if agent.training:
                agent.episode += 1

            self.env.reset()
            self.last_episode_len = self.episode_step

        # Log stats
        sundown = time.time()
        frames = self.episode_step * self.action_repeat

        logs = {'time': sundown - agent.birthday,
                'step': agent.step,
                'frame': agent.step * self.action_repeat,
                'episode': agent.episode,
                'accuracy'if self.suite == 'classify' else 'reward':
                    self.episode_reward / max(1, self.episode_step * self.suite == 'classify'),  # Multi-step/batch acc
                'fps': frames / (sundown - self.daybreak)} if not self.disable \
            else None

        if self.episode_done:
            self.episode_step = self.episode_reward = 0
            self.daybreak = sundown

        return experiences, logs, video_image
