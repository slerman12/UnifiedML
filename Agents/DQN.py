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
    Generalized to continuous action spaces, classification, and generative modeling"""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, classify, generate, device, log,  # On-boarding
                 num_actors=5,  # AC2
                 num_actions=2, num_critics=2):  # DQN
        super().__init__()

        self.discrete = discrete and not generate  # Continuous supported!
        self.supervise = classify  # And classification...
        self.RL = RL
        self.generate = generate  # And generative modeling, too
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps
        self.action_dim = math.prod(obs_shape) if generate else action_shape[-1]

        if not (self.RL or self.generate):
            num_actors = num_actions = 1

        self.num_actions = num_actions if self.RL or self.generate else 1  # Num actions sampled per actor

        self.encoder = CNNEncoder(obs_shape, optim_lr=lr)

        # Continuous actions creator
        self.creator = None if self.discrete \
            else GaussianActorEnsemble(self.encoder.repr_shape, feature_dim, hidden_dim,
                                       self.action_dim, ensemble_size=num_actors,
                                       stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                       optim_lr=lr)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, feature_dim, hidden_dim, self.action_dim,
                                      ensemble_size=num_critics, sigmoid=False, discrete=discrete,
                                      optim_lr=lr, target_tau=target_tau)

        self.actor = CategoricalCriticActor(stddev_schedule)

        # Data augmentation
        self.aug = IntensityAug(0.05) if discrete else RandomShiftsAug(pad=4)

        # Birth

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.creator, self.critic, self.actor):
            obs = torch.as_tensor(obs, device=self.device)

            # "Imagine" / "See"
            obs = torch.randn(self.encoder.repr_dim) if self.generate \
                else self.encoder(obs)

            # "Candidate actions"
            creations = None if self.discrete \
                else self.creator(obs, self.step).sample(self.num_actions) if self.training \
                else self.creator(obs, self.step).mean

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

        # "Envision" / "Perceive"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # Encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # Actor-Critic -> Generator-Discriminator conversion
        if self.generate:
            action = obs.flatten(-3) / 127.5 - 1
            next_obs[:] = float('nan')
            label[:] = float('nan')
            reward[:] = 0

        # "Journal teachings"

        logs = {'time': time.time() - self.birthday,
                'step': self.step, 'episode': self.episode} if self.log \
            else None

        instruction = ~torch.isnan(label)

        # "Acquire Wisdom"

        supervised_loss = critic_loss = 0

        # Classification
        if instruction.any():
            # "Via Example" / "Parental Support" / "School"

            # "Candidate classifications"
            creations = self.creator(obs[instruction], self.step).mean

            # Inference
            y_predicted = self.actor(self.critic(obs[instruction], creations), self.step).best

            mistake = cross_entropy(y_predicted, label[instruction].long(), reduction='none')

            # Supervised learning
            if self.supervise:
                # Supervised loss
                supervised_loss = mistake.mean()

                # Update supervised
                Utils.optimize(supervised_loss,
                               self.creator, retain_graph=True)

                if self.log:
                    logs.update({'supervised_loss': supervised_loss.item()})
                    logs.update({'accuracy': (torch.argmax(y_predicted, -1)
                                              == label[instruction]).float().mean().item()})

            # (Auxiliary) reinforcement
            if self.RL:
                action[instruction] = torch.softmax(y_predicted, -1).detach()
                reward[instruction] = -mistake[:, None].detach()
                next_obs[instruction, :] = float('nan')

        # Reinforcement learning / generative modeling
        if self.RL or self.generate:
            # "Imagine"

            # Generative modeling
            if self.generate:
                obs = torch.randn_like(obs)

                # "Candidate generations"
                creations = self.creator(obs[:len(obs) // 2], self.step).mean

                generated_image = self.actor(self.critic(obs[:len(obs) // 2], creations), self.step).best

                action[:len(obs) // 2] = generated_image
                reward[:len(obs) // 2] = 10  # Discriminate

            # "Discern"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.creator,
                                                      obs, action, reward, discount, next_obs,
                                                      self.step, self.num_actions, logs=logs)

            # Update critic
            Utils.optimize(critic_loss,
                           self.critic)

            self.critic.update_target_params()

        # Update encoder
        self.encoder.optim.step()
        self.encoder.optim.zero_grad(set_to_none=True)

        if self.generate or self.RL and not self.discrete:
            # "Change" / "Grow"

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.creator, self.critic, obs.detach(),
                                                           self.step, self.num_actions, logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.creator)

        return logs
