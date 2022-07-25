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
from Blocks.Actors import EnsembleActor, CategoricalCriticActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning


class DPGAgent(torch.nn.Module):
    """Deep Policy Gradient (DPG)
    Treats discrete RL as continuous RL with optionally one-hots for actions"""
    def __init__(self,
                 obs_shape, action_shape, trunk_dim, hidden_dim, data_stats, recipes,  # Architecture
                 lr, weight_decay, ema_decay, ema,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, supervise, generate, device, parallel, log,  # On-boarding
                 sample_q=False, Q_one_hot=False, Q_one_hot_next=True, policy_one_hot=False, as_continuous=False):  # DPG
        super().__init__()

        self.discrete = discrete and not generate
        self.supervise = supervise
        self.RL = RL
        self.generate = generate
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.episode = 0
        self.explore_steps = explore_steps
        self.ema = ema
        self.action_dim = math.prod(obs_shape) if generate else action_shape[-1]
        self.action_space = torch.arange(self.action_dim, device=self.device)[None, :]

        self.sample_q = sample_q  # Sample Q values from actor prior to sampling a discrete action
        self.Q_one_hot = Q_one_hot and self.discrete  # One-hot discrete actions in Q-Learning
        self.Q_one_hot_next = Q_one_hot_next and self.discrete  # One-hot discrete next-actions in Q-Learning
        self.policy_one_hot = policy_one_hot and self.discrete  # One-hot discrete actions in Policy Learning
        self.as_continuous = as_continuous or not self.discrete  # Treat actions as continuous end-to-end

        self.encoder = Utils.Rand(trunk_dim) if generate \
            else CNNEncoder(obs_shape, data_stats=data_stats, recipe=recipes.encoder, parallel=parallel,
                            lr=lr, weight_decay=weight_decay, ema_decay=ema_decay if ema else None)

        repr_shape = (trunk_dim,) if generate \
            else self.encoder.repr_shape

        self.actor = EnsembleActor(repr_shape, trunk_dim, hidden_dim, self.action_dim, recipes.actor,
                                   ensemble_size=1, stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                   lr=lr, weight_decay=weight_decay, ema_decay=ema_decay if ema else None)

        self.critic = EnsembleQCritic(repr_shape, trunk_dim, hidden_dim, self.action_dim, recipes.critic,
                                      ensemble_size=2, ignore_obs=generate,
                                      lr=lr, weight_decay=weight_decay, ema_decay=ema_decay)

        self.action_selector = CategoricalCriticActor(stddev_schedule)

        # Image augmentation
        self.aug = instantiate(recipes.aug) if recipes.Aug is not None \
            else IntensityAug(0.05) if discrete else RandomShiftsAug(pad=4)

        # Birth

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor):
            obs = torch.as_tensor(obs, device=self.device)

            # EMA targets
            encoder = self.encoder.ema if self.ema and not self.generate else self.encoder
            actor = self.actor.ema if self.ema else self.actor

            # "See"
            obs = encoder(obs)

            Pi = actor(obs, self.step)

            if not self.as_continuous:
                # Samples a discrete action
                Pi = self.action_selector(Pi, self.step,
                                          sample_q=self.sample_q and self.training, action=self.action_space)

            action = Pi.sample() if self.training \
                else Pi.best if not self.as_continuous \
                else Pi.mean

            if self.training:
                self.step += 1

                # Explore phase
                if self.step < self.explore_steps and not self.generate:
                    action = torch.randint(self.action_dim, size=action.shape) if not self.as_continuous \
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

        # Augment and encode
        obs = self.aug(obs)
        obs = self.encoder(obs)

        # Augment and encode future
        if replay.nstep > 0 and not self.generate:
            with torch.no_grad():
                next_obs = self.aug(next_obs)
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

            # Inference
            y_predicted = self.actor(obs[instruction], self.step).mean

            mistake = cross_entropy(y_predicted, label[instruction].long(), reduction='none')

            # Supervised learning
            if self.supervise:
                # Supervised loss
                supervised_loss = mistake.mean()

                # Update supervised
                Utils.optimize(supervised_loss,
                               self.actor, retain_graph=True)

                if self.log:
                    correct = (torch.argmax(y_predicted, -1) == label[instruction]).float()

                    logs.update({'supervised_loss': supervised_loss.item(),
                                 'accuracy': correct.mean().item()})

            # (Auxiliary) reinforcement
            if self.RL:
                half = len(instruction) // 2
                mistake[:half] = cross_entropy(y_predicted[:half].uniform_(-1, 1),
                                               label[instruction][:half].long(), reduction='none')
                action[instruction] = y_predicted.detach()
                reward[instruction] = -mistake[:, None].detach()  # reward = -error
                next_obs[instruction] = float('nan')

        # Reinforcement learning / generative modeling
        if self.RL or self.generate:
            # "Imagine"

            # Generative modeling
            if self.generate:
                half = len(obs) // 2
                generated_image = self.actor(obs[:half], self.step).mean

                action[:half], reward[:half] = generated_image, 0  # Discriminate

            # "Discern"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.actor,
                                                      obs, action, reward, discount, next_obs, self.step,
                                                      one_hot=not self.as_continuous or self.Q_one_hot,
                                                      one_hot_next=self.Q_one_hot_next, logs=logs)

            # Update critic
            Utils.optimize(critic_loss,
                           self.critic)

        # Update encoder
        if not self.generate:
            Utils.optimize(None,  # Using gradients from previous losses
                           self.encoder)

        if self.RL or self.generate:
            # "Change" / "Grow"

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(), self.step,
                                                           one_hot=self.policy_one_hot, logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.actor)

        return logs
