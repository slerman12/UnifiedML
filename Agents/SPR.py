# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.nn.functional import cross_entropy

from Blocks.Architectures import MLP
from Blocks.Architectures.Vision.CNN import CNN
from Blocks.Architectures.Vision.ResNet import MiniResNet

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import EnsembleGaussianActor, CategoricalCriticActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning, SelfSupervisedLearning


class SPRAgent(torch.nn.Module):
    """Self-Predictive Representations (https://arxiv.org/abs/2007.05929)
    Generalized to continuous action spaces, classification, and generative modeling"""
    def __init__(self,
                 obs_spec, action_spec, trunk_dim, hidden_dim, standardize, norm, recipes,  # Architecture
                 lr, lr_decay_epochs, weight_decay, ema_decay, ema,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, supervise, generate, device, parallel, log,  # On-boarding
                 depth=5  # SPR
                 ):
        super().__init__()

        self.discrete = discrete and not generate  # Continuous supported!
        self.supervise = supervise  # And classification...
        self.RL = RL
        self.generate = generate  # And generative modeling, too
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.frame = 0
        self.episode = self.epoch = 1
        self.explore_steps = explore_steps
        self.ema = ema

        self.num_actions = action_spec.num_actions or 1

        self.depth = depth

        # Image augmentation
        self.aug = Utils.instantiate(recipes.aug) or (IntensityAug(0.05) if discrete
                                                      else RandomShiftsAug(pad=4))

        self.encoder = Utils.Rand(trunk_dim) if generate \
            else CNNEncoder(obs_spec, standardize=standardize, norm=norm, **recipes.encoder, parallel=parallel,
                            lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                            ema_decay=ema_decay * ema)

        repr_shape = (trunk_dim, 1, 1) if generate \
            else self.encoder.repr_shape

        # Continuous actions
        self.actor = None if self.discrete \
            else EnsembleGaussianActor(repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.actor,
                                       ensemble_size=1, stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                       lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                       ema_decay=ema_decay * ema)

        # Dynamics
        if not generate:
            shape = [s + self.num_actions if i == 0 else s for i, s in enumerate(repr_shape)]
            resnet = MiniResNet(input_shape=shape, stride=1, dims=(64, repr_shape[0]), depths=(1,))
            obs_spec['obs_shape'] = repr_shape
            # print('k')
            # cnn = CNN(shape, repr_shape[0], stride=1, padding=1)
            # print(cnn)

            self.dynamics = CNNEncoder(obs_spec, self.num_actions,
                                       eyes=torch.nn.Sequential(resnet, Utils.ShiftMaxNorm(-3)),
                                       # eyes=cnn,
                                       lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay)

            # Self supervisors
            self.projector = CNNEncoder(obs_spec,
                                        eyes=MLP(self.encoder.repr_shape, hidden_dim, hidden_dim, 2),
                                        lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                        ema_decay=ema_decay)
            obs_spec['obs_shape'] = [hidden_dim, 1, 1]

            self.predictor = CNNEncoder(obs_spec,
                                        eyes=MLP(self.encoder.repr_shape, hidden_dim, hidden_dim, 2),
                                        lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay)

        self.critic = EnsembleQCritic(repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.critic,
                                      ensemble_size=2, discrete=self.discrete, ignore_obs=generate,
                                      lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                      ema_decay=ema_decay)

        self.action_selector = CategoricalCriticActor(stddev_schedule)

        # Birth

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor, self.critic):
            obs = torch.as_tensor(obs, device=self.device).float()

            # EMA shadows
            encoder = self.encoder.ema if self.ema and not self.generate else self.encoder
            actor = self.actor.ema if self.ema and not self.discrete else self.actor
            critic = self.critic.ema if self.ema else self.critic

            # "See"
            obs = encoder(obs)

            Pi = self.action_selector(critic(obs), self.step) if self.discrete \
                else actor(obs, self.step)

            action = Pi.sample() if self.training \
                else Pi.best if self.discrete \
                else Pi.mean

            if self.training:
                self.step += 1
                self.frame += len(obs)

                # Explore phase
                if self.step < self.explore_steps and not self.generate:
                    action = torch.randint(self.num_actions, size=action.shape) if self.discrete \
                        else action.uniform_(-1, 1)

            return action

    # "Dream"
    def learn(self, replay):
        # "Recollect"

        batch = replay.sample(True)
        obs, action, reward, discount, next_obs, label, *traj, step, ids, meta = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r, traj_l = traj

        # "Envision" / "Perceive"

        # Augment
        obs = self.aug(obs)

        # Actor-Critic -> Generator-Discriminator conversion
        if self.generate:
            obs = (obs - self.low) * 2 / (self.high - self.low) - 1  # Normalize first
            action, reward[:] = obs.flatten(-3), 1
            next_obs[:] = label[:] = float('nan')

        # Encode
        features = self.encoder(obs, pool=False)
        obs = self.encoder.pool(features)

        # Augment and encode future
        if replay.nstep > 0 and not self.generate:
            with torch.no_grad():
                next_obs = self.aug(next_obs)
                next_obs = self.encoder(next_obs)

        # "Journal teachings"

        offline = replay.offline

        logs = {'time': time.time() - self.birthday, 'step': self.step + offline, 'frame': self.frame + offline,
                'epoch' if offline else 'episode':  self.epoch if offline else self.episode} if self.log \
            else None

        if offline:
            self.step += 1
            self.frame += len(obs)
            self.epoch = replay.epoch

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
                               self.actor, epoch=self.epoch if offline else self.episode, retain_graph=True)

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

            # "Predict" / "Plan" / "Discern"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.actor,
                                                      obs, action, reward, discount, next_obs,
                                                      self.step, logs=logs)

            # Dynamics loss
            dynamics_loss = 0 if replay.nstep == 0 or self.generate \
                else SelfSupervisedLearning.dynamicsLearning(features, traj_o, traj_a, traj_r,
                                                             self.encoder, self.dynamics, self.projector, self.predictor,
                                                             depth=min(replay.nstep, self.depth), logs=logs)

            # Update critic, dynamics
            Utils.optimize(critic_loss + dynamics_loss,
                           self.critic,
                           self.dynamics, self.projector, self.predictor, epoch=self.epoch if offline else self.episode)

        # Update encoder
        if not self.generate:
            Utils.optimize(None,  # Using gradients from previous losses
                           self.encoder, epoch=self.epoch if offline else self.episode)

        if self.generate or self.RL and not self.discrete:
            # "Change" / "Grow"

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                           self.step, logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.actor, epoch=self.epoch if offline else self.episode)

        return logs
