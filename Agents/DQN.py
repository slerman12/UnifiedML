# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
import math

from hydra.utils import instantiate

import torch
from torch.nn.functional import cross_entropy

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import EnsembleGaussianActor, CategoricalCriticActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning


class DQNAgent(torch.nn.Module):
    """Deep Q Network
    Generalized to continuous action spaces, classification, and generative modeling"""
    def __init__(self,
                 obs_shape, action_shape, trunk_dim, hidden_dim, recipes,  # Architecture
                 lr, ema_tau, ema,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, supervise, generate, device, log,  # On-boarding
                 num_actions=2, num_critics=2):  # DQN
        super().__init__()

        self.discrete = discrete and not generate  # Continuous supported!
        self.supervise = supervise  # And classification...
        self.RL = RL
        self.generate = generate  # And generative modeling, too
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps
        self.ema = ema

        self.action_dim = math.prod(obs_shape) if generate else action_shape[-1]

        self.num_actions = num_actions  # Num actions sampled per actor

        self.encoder = Utils.Rand(trunk_dim) if generate \
            else CNNEncoder(obs_shape, recipe=recipes.encoder, optim_lr=lr, ema_tau=ema_tau if ema else None)

        feature_shape = (trunk_dim,) if generate else self.encoder.feature_shape

        # Continuous actions
        self.actor = None if self.discrete \
            else EnsembleGaussianActor(feature_shape, trunk_dim, hidden_dim, self.action_dim, recipes.actor,
                                       ensemble_size=1,
                                       stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                       optim_lr=lr, ema_tau=ema_tau if ema else None)

        self.critic = EnsembleQCritic(feature_shape, trunk_dim, hidden_dim, self.action_dim, recipes.critic,
                                      ensemble_size=num_critics, discrete=self.discrete, ignore_obs=generate,
                                      optim_lr=lr, ema_tau=ema_tau)

        self.action_selector = CategoricalCriticActor(stddev_schedule)

        # Data augmentation
        self.aug = instantiate(recipes.aug) if recipes.Aug is not None \
            else IntensityAug(0.05) if discrete else RandomShiftsAug(pad=4)

        # Birth

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor, self.critic, self.actor):
            obs = torch.as_tensor(obs, device=self.device)

            # EMA targets
            encoder = self.encoder.ema if self.ema else self.encoder
            actor = self.actor.ema if self.ema else self.actor

            # "See"
            obs = encoder(obs)

            actions = None if self.discrete \
                else actor(obs, self.step).sample(self.num_actions) if self.training \
                else actor(obs, self.step).mean

            # DQN action selector is based on critic
            Pi = self.action_selector(self.critic(obs, actions), self.step)

            action = Pi.sample() if self.training \
                else Pi.best

            if self.training:
                self.step += 1

                # Explore phase
                if self.step < self.explore_steps and not self.generate:
                    action = torch.randint(self.action_dim, size=action.shape) if self.discrete \
                        else action.uniform_(-1, 1)

            return action

    # "Dream"
    def learn(self, replay):
        # "Recollect"

        batch = next(replay)
        obs, action, reward, discount, next_obs, label, *traj, step = Utils.to_torch(
            batch, self.device)

        # Actor-Critic -> Generator-Discriminator conversion
        if self.generate:
            action, reward[:] = obs.flatten(-3) / 127.5 - 1, 1
            next_obs[:] = label[:] = float('nan')

        # "Envision" / "Perceive"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # Encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # "Journal teachings"

        logs = {'time': time.time() - self.birthday,
                'step': self.step, 'episode': self.episode} if self.log \
            else None

        instruction = ~torch.isnan(label)

        # "Acquire Wisdom"

        # Classification
        if instruction.any():
            # "Via Example" / "Parental Support" / "School"

            y_actual = label[instruction].long()

            # Supervised learning
            if self.supervise:
                y_predicted = self.actor(obs[instruction], self.step).mean[:, 0]

                # Supervised loss
                supervised_loss = cross_entropy(y_predicted, y_actual)  # view/flatten/repeat_interleave

                # Update supervised
                Utils.optimize(supervised_loss,
                               self.actor, retain_graph=self.RL)

                if self.log:
                    logs.update({'supervised_loss': supervised_loss.item()})
                    logs.update({'accuracy': (torch.argmax(y_predicted, -1)
                                              == y_actual).float().mean().item()})

            # (Auxiliary) reinforcement
            if self.RL:
                actions = self.actor(obs[instruction], self.step).rsample(self.num_actions)

                y_predicted = self.action_selector(self.critic(obs[instruction], actions), self.step).best

                half = len(y_predicted) // 2
                y_predicted[:half].uniform_()

                mistake = cross_entropy(y_predicted, y_actual, reduction='none')

                reward[instruction] = -mistake[:, None].detach()
                action[instruction], next_obs[instruction] = y_predicted.softmax(-1).detach(), float('nan')

        # Reinforcement learning / generative modeling
        if self.RL or self.generate:
            # "Imagine"

            # Generative modeling
            if self.generate:
                next_obs[:] = float('nan')
                actions = self.actor(obs[:len(obs) // 2], self.step).mean

                generated_image = self.action_selector(self.critic(obs[:len(obs) // 2], actions), self.step).best

                half = len(obs) // 2
                action[:half], reward[:half] = generated_image, 0  # Discriminate

            # "Discern"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.actor,
                                                      obs, action, reward, discount, next_obs,
                                                      self.step, self.num_actions, logs=logs)

            # Update critic
            Utils.optimize(critic_loss,
                           self.critic)

        # Update encoder
        if not self.generate:
            Utils.optimize(None,  # Using gradients from previous losses
                           self.encoder)

        if self.generate or self.RL and not self.discrete:
            # "Change" / "Grow"

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                           self.step, self.num_actions, logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.actor)

        return logs
