# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.nn.functional import cross_entropy

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import EnsemblePiActor, CategoricalCriticActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning

# TODO replay spec -> dict, creator/store, bluehive memory, debug log_video, AC2, ViT & architectures!
class DQNAgent(torch.nn.Module):
    """Deep Q Network
    Generalized to continuous action spaces, classification, and generative modeling"""
    def __init__(self,
                 obs_spec, action_spec, num_actions, trunk_dim, hidden_dim, standardize, norm, recipes,  # Architecture
                 lr, lr_decay_epochs, weight_decay, ema_decay, ema,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, supervise, generate, device, parallel, log,  # On-boarding
                 num_critics=2  # DQN
                 ):
        super().__init__()

        self.discrete = discrete and not generate  # Continuous supported!
        self.supervise = supervise  # And classification...
        self.RL = RL or generate
        self.generate = generate  # And generative modeling, too
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.frame = 0
        self.episode = self.epoch = 1
        self.explore_steps = explore_steps
        self.ema = ema

        self.num_actions = num_actions

        # Image augmentation TODO classify differs due to image aug!! - apples/oranges
        self.aug = Utils.instantiate(recipes.aug) or (torch.nn.Sequential(RandomShiftsAug(pad=4),
                                                                          IntensityAug(0.05)) if action_spec.discrete
                                                      else RandomShiftsAug(pad=4))

        # RL -> generate conversion
        if self.generate:
            standardize = False
            norm = True  # Normalize Obs to range [-1, 1]

            # Action = Imagined Obs
            action_spec.update({'shape': obs_spec.shape, 'discrete_bins': None,
                                'low': -1, 'high': 1, 'discrete': False})

            # Remove encoder, replace trunk with random noise
            recipes.encoder.Eyes = torch.nn.Identity()  # Generate "imagines" — no need for "seeing" with Eyes
            recipes.actor.trunk = Utils.Rand(size=trunk_dim)  # Generator observes random Gaussian noise as input

        # # Continuous -> discrete conversion
        if self.discrete and not action_spec.discrete:
            assert self.num_actions > 1, 'Num actions cannot be 1 when discrete; try the "num_actions=" flag (>1) to ' \
                                         'divide each action dimension into discrete bins, or specify "discrete=false".'

            action_spec.discrete_bins = self.num_actions  # Continuous env has no discrete bins by default, must specify

        self.encoder = CNNEncoder(obs_spec, standardize=standardize, norm=norm, **recipes.encoder, parallel=parallel,
                                  lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                  ema_decay=ema_decay * ema)

        self.actor = EnsemblePiActor(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.actor,
                                     ensemble_size=num_critics if self.discrete and self.RL else 1,
                                     discrete=self.discrete, stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                     lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                     ema_decay=ema_decay * ema)

        # When discrete, Critic <- Actor
        if self.discrete:
            recipes.critic.trunk = self.actor.trunk
            recipes.critic.Q_head = self.actor.Pi_head.ensemble

        self.critic = EnsembleQCritic(self.encoder.repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.critic,
                                      ensemble_size=num_critics if self.RL else 1,  # todo discrete here?
                                      discrete=self.discrete, ignore_obs=self.generate,
                                      lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                      ema_decay=ema_decay)

        self.action_selector = CategoricalCriticActor(stddev_schedule)

        # Birth

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor, self.critic):
            obs = torch.as_tensor(obs, device=self.device).float()

            # Exponential moving average (EMA) shadows
            encoder = self.encoder.ema if self.ema else self.encoder
            actor = self.actor.ema if self.ema else self.actor
            critic = self.critic.ema if self.ema else self.critic

            # See
            obs = encoder(obs)

            # Act
            Pi = actor(obs, self.step)

            action = Pi.sample(self.num_actions) if self.training \
                else Pi.best if self.discrete \
                else Pi.mean

            if self.training:
                # Select among candidate actions based on Q-value
                if self.num_actions > 1:
                    All_Qs = getattr(Pi, 'All_Qs', None)  # Discrete Actor policy already knows all Q-values

                    action = self.action_selector(critic(obs, action, All_Qs), self.step, action).best

                self.step += 1
                self.frame += len(obs)

                if self.step < self.explore_steps and not self.generate:
                    # Explore
                    action.uniform_(actor.low or 1, actor.high or 9)  # Env will automatically round if discrete

            return action

    # "Dream"
    def learn(self, replay):
        # "Recollect"

        batch = next(replay)
        obs, action, reward, discount, next_obs, label, step, ids, meta = Utils.to_torch(
            batch, self.device)

        # "Envision" / "Perceive"

        # Augment, encode present
        obs = self.aug(obs)
        obs = self.encoder(obs)

        if replay.nstep > 0 and not self.generate:
            with torch.no_grad():
                # Augment, encode future
                next_obs = self.aug(next_obs)
                next_obs = self.encoder(next_obs)

        # TODO maybe action = obs for this and if imitate: action = label here, the rest below if generate or imitate
        # Actor-Critic -> Generator-Discriminator conversion  Todo move below?
        if self.generate:
            action, reward[:] = obs, 1  # "Real"
            next_obs[:] = label[:] = float('nan')

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
            Pi = self.actor(obs)

            y_predicted = (Pi.All_Qs if self.discrete else Pi.mean).mean(1)  # Average over ensembles

            mistake = cross_entropy(y_predicted, label.long(), reduction='none')
            correct = (y_predicted.argmax(1) == label).float()
            accuracy = correct.mean()

            if self.log:
                logs.update({'accuracy': accuracy})

            # Supervised learning
            if self.supervise:
                # Supervised loss
                supervised_loss = mistake.mean()

                # Update supervised
                Utils.optimize(supervised_loss,
                               self.actor, epoch=self.epoch if offline else self.episode, retain_graph=True)

                if self.log:
                    logs.update({'supervised_loss': supervised_loss})

            # (Auxiliary) reinforcement
            if self.RL:
                half = len(obs) // 2
                mistake[:half] = cross_entropy(y_predicted[:half].uniform_(-1, 1),
                                               label[:half].long(), reduction='none')
                action = (y_predicted.argmax(1, keepdim=True) if self.discrete else y_predicted).detach()
                reward = -mistake.detach()  # reward = -error
                next_obs[:] = float('nan')

                if self.log:
                    logs.update({'reward': reward})

        # Reinforcement learning / generative modeling
        if self.RL:
            # "Imagine"

            # Generative modeling
            if self.generate:
                half = len(obs) // 2

                Pi = self.actor(obs[:half])
                generated_image = Pi.mean.flatten(1)

                action[:half], reward[:half] = generated_image, 0  # Discriminate "fake"

            # "Discern"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.actor,
                                                      obs, action, reward, discount, next_obs,
                                                      self.step, self.num_actions, logs=logs)

            # Update critic
            Utils.optimize(critic_loss,
                           self.critic, epoch=self.epoch if offline else self.episode)

        # Update encoder
        Utils.optimize(None,  # Using gradients from previous losses
                       self.encoder, epoch=self.epoch if offline else self.episode)

        if self.RL and not self.discrete:
            # "Change" / "Grow"

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                           self.step, self.num_actions, logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.actor, epoch=self.epoch if offline else self.episode)

        return logs
