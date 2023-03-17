# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.nn.functional import cross_entropy

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder, ResidualBlockEncoder
from Blocks.Actors import GaussianActorEnsemble, CategoricalCriticActor, SGAActor
from Blocks.Critics import EnsembleQCritic
from Blocks.Architectures.MLP import MLPBlock

from Losses import QLearning, PolicyLearning, SelfSupervisedLearning


class AscendAgent(torch.nn.Module):
    """A Critic Enjoying No Directives (Ascend) Agent"""
    # TODO convert to DPG but with num_actions (DPG-AC2)
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, target_tau,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, device, log,  # On-boarding
                 critic_ensemble_size=3, num_actions=2, steps=15  # Ascend
                 ):
        super().__init__()

        self.discrete = discrete
        self.RL = RL
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps

        self.num_actions = num_actions

        # Models
        self.encoder = CNNEncoder(obs_shape, optim_lr=lr, target_tau=target_tau).to(device)

        if not discrete:
            self.creator = SGAActor(self.critic, steps=steps)

        self.critic = EnsembleQCritic(self.encoder.repr_shape, feature_dim, hidden_dim, action_shape[-1],
                                      critic_ensemble_size, discrete=False,  # False for now
                                      optim_lr=lr, target_tau=target_tau).to(device)

        self.actor = CategoricalCriticActor(stddev_schedule)

        self.dynamics = ResidualBlockEncoder(self.encoder.repr_shape, action_shape[-1],
                                             pixels=False, isotropic=True,
                                             optim_lr=lr).to(device)

        self.projector = MLPBlock(self.encoder.flattened_dim, hidden_dim, hidden_dim, hidden_dim,
                                  depth=2, layer_norm=True,
                                  target_tau=target_tau, optim_lr=lr).to(device)

        self.obs_predictor = MLPBlock(hidden_dim, hidden_dim, hidden_dim, hidden_dim,
                                      depth=2, layer_norm=True,
                                      optim_lr=lr).to(device)
        self.reward_predictor = MLPBlock(hidden_dim, 1, hidden_dim, hidden_dim,
                                         depth=2, layer_norm=True,
                                         optim_lr=lr).to(device)

        # Data augmentation
        self.aug = IntensityAug(0.05) if self.discrete else RandomShiftsAug(pad=4)

        # Birth

    # "Play"
    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)

            # "See"
            obs = self.encoder(obs)

            # "Candidate actions"
            creation = None if self.discrete \
                else self.creator(obs).sample(self.num_actions)

            # DQN actor is based on critic
            Pi = self.actor(self.critic(obs, creation), self.step)

            action = Pi.sample() if self.training else Pi.best

            if self.training:
                self.step += 1

                # Explore phase
                if self.step < self.explore_steps and self.training:
                    action = torch.randint(self.actor.action_dim, size=action.shape) if self.discrete \
                        else action.uniform_(-1, 1)

            return action

    # "Dream"
    def update(self, replay):
        # "Recollect"

        batch = next(replay)
        obs, action, reward, discount, next_obs, label, *traj, step = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r, traj_l = traj

        # "Imagine" / "Envision"

        # Augment
        obs = self.aug(obs)
        next_obs = self.aug(next_obs)

        # Encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # "Journal teachings"

        logs = {'episode': self.episode, 'step': self.step} if self.log else None

        instruction = ~torch.isnan(label.flatten(-1).sum(1))

        # "Acquire Wisdom"

        if instruction.any():
            # "Via Example" / "Parental Support" / "School"

            # "Candidate actions"
            creations = None if self.discrete \
                else self.critic(self.creator(obs[instruction]).sample(self.num_actions))

            # Infer
            action = self.actor(self.critic(obs[instruction], creations), self.step)

            mistake = cross_entropy(action, label[instruction], reduction='none')

            # Supervised loss
            supervised_loss = mistake.mean()

            if self.log:
                logs.update({'supervised_loss': supervised_loss.item()})

            # Update actor
            Utils.optimize(supervised_loss,
                           self.encoder,
                           self.actor)

            if self.RL:
                # Auxiliary reinforcement
                reward[instruction] = -mistake

        if self.RL:
            # "Predict" / "Discern" / "Learn" / "Grow"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.actor if self.discrete else self.creator,
                                                      self.critic, obs, action, reward, discount, next_obs,
                                                      self.step, logs=logs)

            # Convert discrete action trajectories to one-hot
            if self.discrete:
                traj_a = Utils.one_hot(traj_a, num_classes=self.actorSAURUS.action_dim)

            future = -torch.isnan(next_obs.flatten(1).sum(1))

            # Dynamics loss
            dynamics_loss = SelfSupervisedLearning.dynamicsLearning(
                obs[future], traj_o[future], traj_a[future], traj_r[future], self.encoder, self.dynamics,
                self.projector, self.obs_predictor, self.reward_predictor, self.depth, logs
            )

            # Update critic, dynamics
            Utils.optimize(critic_loss + dynamics_loss,
                           self.encoder,
                           self.critic,
                           self.dynamics, self.projector, self.obs_predictor, self.reward_predictor)

            # Actor loss
            if not self.discrete:
                actor_loss = PolicyLearning.deepPolicyGradient(self.creator, self.critic, obs.detach(),
                                                               self.step, logs=logs)

                # Update actor
                Utils.optimize(actor_loss,
                               self.actor)
        return logs