# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time
import math

import torch
from torch.nn.functional import cross_entropy

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import GaussianActorEnsemble, CategoricalCriticActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning


class DQNAgent(torch.nn.Module):
    """Deep Q Network
    Generalized to continuous action spaces and classification"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, classify, generate, device, log,  # On-boarding
                 num_actors=5, num_actions=2):  # DQN (for non-discrete support)
        super().__init__()

        self.discrete = discrete and not generate  # Continuous supported!
        self.RL = RL
        self.classify = classify  # And classification...
        self.generate = generate  # And generative modeling, too
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps
        self.action_dim = math.prod(obs_shape) if generate else action_shape[-1]

        self.num_actions = num_actions  # Num actions sampled per actor

        self.encoder = CNNEncoder(obs_shape, optim_lr=lr)

        # Continuous actions creator
        self.creator = None if self.discrete \
            else GaussianActorEnsemble(self.encoder.repr_shape, feature_dim, hidden_dim,
                                       self.action_dim, ensemble_size=num_actors,
                                       stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                       optim_lr=lr)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, feature_dim, hidden_dim, self.action_dim,
                                      sigmoid=generate, discrete=discrete,
                                      optim_lr=lr, target_tau=target_tau)

        self.actor = CategoricalCriticActor(stddev_schedule)

        # Data augmentation
        self.aug = IntensityAug(0.05) if discrete else RandomShiftsAug(pad=4)

        # Birth

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.creator, self.critic, self.actor):
            obs = torch.as_tensor(obs, device=self.device)

            # "See"
            obs = self.encoder(obs)

            # "Candidate actions"
            creations = None if self.discrete \
                else self.creator(obs, self.step).sample(self.num_actions)

            # DQN actor is based on critic
            Pi = self.actor(self.critic(obs, creations), self.step)

            action = Pi.sample() if self.training \
                else Pi.best

            if self.training:
                self.step += 1

                # Explore phase  TODO test without
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

        # "Envision"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # Actor-Critic -> Generator-Discriminator conversion
        if self.generate:
            action = obs.clone().flatten(-3)
            obs._uniform()
            next_obs[:] = label[:] = float('nan')
            reward[:] = 0

        # "Journal teachings"

        logs = {'time': time.time() - self.birthday,
                'step': self.step, 'episode': self.episode} if self.log \
            else None

        instruction = ~torch.isnan(label)

        # "Acquire Wisdom"

        # Supervised learning
        if instruction.any():
            # "Via Example" / "Parental Support" / "School"

            # Supervised input
            x = self.encoder(obs)

            # "Candidate classifications"
            creations = self.creator(x[instruction], self.step).mean

            # Inference
            y_predicted = self.actor(self.critic(x[instruction], creations), self.step).best if self.RL \
                else creations[:, torch.randint(high=creations.shape[1] - 1, size=[1]).data]

            mistake = cross_entropy(y_predicted, label[instruction].long(), reduction='none')

            if self.classify:
                # Supervised loss
                supervised_loss = mistake.mean()

                # Update supervised
                Utils.optimize(supervised_loss,
                               self.encoder,
                               self.creator)

                if self.log:
                    logs.update({'supervised_loss': supervised_loss.item()})
                    logs.update({'accuracy': (torch.argmax(y_predicted, -1)
                                              == label[instruction]).float().mean().item()})

            # (Auxiliary) reinforcement
            if self.RL:
                action[instruction] = y_predicted.detach()
                reward[instruction] = -mistake[:, None].detach()
                next_obs[instruction, :] = float('nan')

        # Reinforcement learning / Generative Modeling
        if self.RL or self.generate:
            # "Perceive"

            # Encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

            # "Imagine"

            # Generative modeling
            if self.generate:
                # "Candidate generations"
                creations = self.creator(obs[:len(obs) // 2], self.step).mean

                generated_image = self.actor(self.critic(obs[:len(obs) // 2], creations), self.step).best
                generated_image = torch.round(generated_image * 255 + 255) / 2

                action[:len(obs) // 2] = generated_image
                reward[:len(obs) // 2] = 1  # Discriminate

            # "Predict" / "Discern" / "Learn" / "Grow"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.creator,
                                                      obs, action, reward, discount, next_obs,
                                                      self.step, self.num_actions, logs=logs)

            # Update critic
            Utils.optimize(critic_loss,
                           self.encoder,
                           self.critic)

            self.critic.update_target_params()

            if not self.discrete:
                # Actor loss
                actor_loss = PolicyLearning.deepPolicyGradient(self.creator, self.critic, obs.detach(),
                                                               self.step, self.num_actions, logs=logs)

                # Update actor
                Utils.optimize(actor_loss,
                               self.creator)

        return logs
