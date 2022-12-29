# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.nn.functional import cross_entropy

import Utils

from Blocks.Augmentations import RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import EnsemblePiActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning


class DrQV2Agent(torch.nn.Module):
    """Data Regularized Q-Learning version 2 (https://arxiv.org/abs/2107.09645)
    Generalized to discrete action spaces and classify tasks"""
    def __init__(self,
                 obs_spec, action_spec, num_actions, trunk_dim, hidden_dim, standardize, norm, recipes,  # Architecture
                 lr, lr_decay_epochs, weight_decay, ema_decay, ema,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, supervise, generate, device, parallel, log,  # On-boarding
                 ):
        super().__init__()

        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.frame = 0
        self.episode = self.epoch = 1
        self.explore_steps = explore_steps

        # Continuous RL
        assert not discrete and RL, 'DrQV2Agent only supports continuous RL. Set "discrete=false RL=true".'
        assert not generate, 'DrQV2Agent does not support generative modeling.'

        # Image augmentation
        self.aug = Utils.instantiate(recipes.aug) or RandomShiftsAug(pad=4)

        # Discrete -> continuous conversion
        if action_spec.discrete:
            # Normalizing actions to range [-1, 1] significantly helps continuous RL
            action_spec.low, action_spec.high = (-1, 1)

        self.encoder = CNNEncoder(obs_spec, standardize=standardize, **recipes.encoder, parallel=parallel, lr=lr)

        self.actor = EnsemblePiActor(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.actor,
                                     ensemble_size=1, stddev_schedule=stddev_schedule, stddev_clip=stddev_clip, lr=lr)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.critic,
                                      ensemble_size=2, lr=lr, ema_decay=ema_decay)

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device).float()

            obs = self.encoder(obs)

            Pi = self.actor(obs, self.step)
            action = Pi.sample() if self.training else Pi.mean

            if self.training:

                self.step += 1
                self.frame += len(obs)

                if self.step < self.explore_steps:
                    # Explore
                    action.uniform_(self.actor.low, self.actor.high)

            # If environment's action space is discrete, environment will argmax action
            return action, {}

    def learn(self, replay):

        # Online RL
        assert not replay.offline, 'DrQV2Agent does not support offline learning. Set "offline=false" or "online=true".'

        batch = next(replay)
        obs, action, reward, discount, next_obs, label, *_ = Utils.to_torch(
            batch, self.device)

        # Supervised -> RL conversion
        instruct = ~torch.isnan(label)

        if instruct.any():
            reward = -cross_entropy(action.squeeze(1), label.long(), reduction='none')  # reward = -error

        logs = {'time': time.time() - self.birthday, 'step': self.step, 'frame': self.frame,
                'episode':  self.episode} if self.log else None

        # Can't do offline here because action is forever NaN in classify
        # logs = {'time': time.time() - self.birthday, 'step': self.step, 'frame': self.frame,
        #         'epoch': self.epoch, 'episode': self.episode} if self.log else None
        #
        # # Online -> Offline conversion
        # if replay.offline:
        #     self.step += 1
        #     self.frame += len(obs)
        #     logs['step'] = self.step
        #     logs['frame'] += 1
        #     self.epoch = replay.epoch

        # Augment, encode present
        obs = self.aug(obs)
        obs = self.encoder(obs)

        if replay.nstep:
            with torch.no_grad():
                # Augment, encode future
                next_obs = self.aug(next_obs)
                next_obs = self.encoder(next_obs)

        # Critic loss
        critic_loss = QLearning.ensembleQLearning(self.critic, self.actor, obs, action, reward, discount, next_obs,
                                                  self.step, logs=logs)

        # Update encoder and critic
        Utils.optimize(critic_loss, self.encoder, self.critic)

        # Actor loss
        actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(), self.step, logs=logs)

        # Update actor
        Utils.optimize(actor_loss, self.actor)

        return logs
