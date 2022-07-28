# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
from math import inf

# from Utils import instantiate, to_torch
from hydra.utils import instantiate


class Environment:
    def __init__(self, env, suite='DMC', task='cheetah_run', frame_stack=1, truncate_episode_steps=1e3, action_repeat=1,
                 offline=False, generate=False, train=True, seed=0, **kwargs):
        self.suite = suite.lower()
        self.offline = offline
        self.generate = generate

        # Offline and generate don't use training rollouts!
        self.disable = (offline or generate) and train

        self.truncate_after = train and truncate_episode_steps or inf  # Truncate episodes shorter (inf if None)

        if not self.disable:
            self.env = instantiate(env, task=task, frame_stack=frame_stack, action_repeat=action_repeat,
                                   offline=offline, generate=generate, train=train, seed=seed, **kwargs)
            self.env.reset()

        self.action_repeat = getattr(getattr(self, 'env', 1), 'action_repeat', 1)  # Optional, can skip frames

        self.episode_done = self.episode_step = self.episode_frame = self.last_episode_len = self.episode_reward = 0
        self.daybreak = None

    def rollout(self, agent, steps=inf, vlog=False):
        if self.daybreak is None:
            self.daybreak = time.time()  # "Daybreak" for whole episode

        experiences = []
        video_image = []

        self.episode_done = self.disable

        step = frame = 0
        while not self.episode_done and step < steps:
            exp = self.env.exp

            # Frame-stacked obs
            obs = getattr(self.env, 'frame_stack', lambda x: x)(exp.obs)

            # Act
            action = agent.act(obs)

            if not self.generate:
                exp = self.env.step(action.cpu().numpy())  # Experience  TODO, agent.stddev?, no cpu in suite

            exp.step = agent.step  # TODO then don't need this, can do exp.action = action?
            experiences.append(exp)

            if vlog or self.generate:
                image_frame = action[:24].view(-1, *exp.obs.shape[1:]) if self.generate \
                    else self.env.render()
                video_image.append(image_frame)

            step += 1
            frame += len(action)

            # Tally reward, done
            self.episode_reward += exp.reward.mean()
            self.episode_done = self.env.episode_done or self.episode_step > self.truncate_after - 2 or self.generate

            if self.env.episode_done:
                self.env.reset()

        agent.episode += agent.training * self.episode_done  # Increment agent episode

        # Tally time
        self.episode_step += step
        self.episode_frame += frame

        if self.episode_done:
            self.last_episode_len = self.episode_step

        # Log stats
        sundown = time.time()
        frames = self.episode_frame * self.action_repeat

        logs = {'time': sundown - agent.birthday,
                'step': agent.step,
                'frame': agent.frame * self.action_repeat,
                'epoch' if self.offline or self.generate else 'episode':
                    (self.offline or self.generate) and agent.epoch or agent.episode,
                'accuracy' if self.suite == 'classify' else 'reward':
                    self.episode_reward / max(1, self.episode_step * self.suite == 'classify'),  # Accuracy is %
                'fps': frames / (sundown - self.daybreak)} if not self.disable \
            else None

        if self.episode_done:
            self.episode_step = self.episode_frame = self.episode_reward = 0
            self.daybreak = sundown

        return experiences, logs, video_image


# Creator - environment
# def create(action, agent, env):
#     if agent.discrete or not (env.action_spec['discrete'] or agent.training):
#         return action  # Discrete -> Discrete, Discrete -> Continuous, or Continuous -> Continuous
#
#     Qs = action.flatten(-2)
#     print(Qs.shape)
#     actions = to_torch(([range(Qs.shape[-1])],), device=Qs.device)[0]
#
#     return getattr(agent, 'creator', agent.action_selector)(actions, agent.step, Qs).sample()  # Continuous -> Discrete


# Creator - environment  TODO move to creator? No cretaor, suite, takes step as arg, samples or argmaxes!
# def create(Qs, step, action=None): # Qs first
#     Qs = Qs.flatten(2)
#
#     if action is None:
#         action = to_torch(([range(Qs.shape[-1])],), device=Qs.device)[0]
