# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch
from torch.nn.functional import cross_entropy
# from torch.nn.functional import cross_entropy, mse_loss as mse

import Utils

from Blocks.Augmentations import IntensityAug, RandomShiftsAug
from Blocks.Encoders import CNNEncoder
from Blocks.Actors import EnsembleActor, CategoricalCriticActor
from Blocks.Critics import EnsembleQCritic

from Losses import QLearning, PolicyLearning


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
        # self.classify = action_spec.discrete  # Including classification and regression...
        self.RL = RL
        # self.imitate = imitate
        self.generate = generate  # And generative modeling, too  # TODO Can be discrete, don't ignore obs, num_acts=255
        self.device = device
        self.log = log
        self.birthday = time.time()
        self.step = self.frame = 0
        self.episode = self.epoch = 1
        self.explore_steps = explore_steps
        self.ema = ema

        self.num_actions = num_actions or action_spec.num_actions

        if self.discrete:
            assert self.num_actions > 1, 'Num actions cannot be 1 when calling continuous env as discrete, ' \
                                         'specify "+agent.num_actions=" flag >1'
            action_spec.num_actions = self.num_actions  # Continuous -> discrete conversion

        if generate:
            # action_spec.num_actions = None
            action_spec.shape = obs_spec.shape
            action_spec.low, action_spec.high = -1, 1

            recipes.encoder.Eyes = Utils.Rand(trunk_dim)

            # if self.discrete:
            #     action_spec.num_actions = 255  # TODO If obs_spec.discrete else num_actions or 10; Need to sample Actor!
            #     action_spec.discrete = False

        # Data stats
        self.low, self.high = obs_spec.low, obs_spec.high

        self.encoder = CNNEncoder(obs_spec, standardize=standardize, norm=norm, **recipes.encoder, parallel=parallel,
                                  lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                  ema_decay=ema_decay * ema)

        repr_shape = (trunk_dim,) if generate \
            else self.encoder.repr_shape

        self.actor = EnsembleActor(repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.actor,
                                   ensemble_size=num_critics if self.discrete and self.RL else 1,  # Num actors
                                   stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                   lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                   ema_decay=ema_decay * ema)

        # Critic <- Actor
        # (Technically, the actor doesn't need to be the critic; they could both update the encoder.
        # Any number of the critics can be shared with any equal number of the actors)
        if self.discrete:
            recipes.critic.trunk = self.actor.trunk
            recipes.critic.Q_head = self.actor.Pi_head.ensemble

            if not self.RL:
                num_critics = 1  # Num actors

        self.critic = EnsembleQCritic(repr_shape, trunk_dim, hidden_dim, action_spec, **recipes.critic,
                                      ensemble_size=num_critics, discrete=self.discrete, ignore_obs=generate,
                                      lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                      ema_decay=ema_decay)

        self.action_selector = CategoricalCriticActor(stddev_schedule)

        # Image augmentation
        self.aug = Utils.instantiate(recipes.aug) or (IntensityAug(0.05) if discrete
                                                      else RandomShiftsAug(pad=4))

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
                    All_Qs = getattr(Pi, 'All_Qs', None)  # Discrete Actor policy knows all Q-values

                    action = self.action_selector(critic(obs, action, All_Qs), self.step).best

                self.step += 1
                self.frame += len(obs)

                # "Explore phase"

                if self.step < self.explore_steps and not self.generate:
                    action = action.uniform_(actor.low, actor.high)  # Env automatically rounds if discrete

            return action

    # "Dream"
    def learn(self, replay):
        # "Recollect"

        batch = next(replay)
        obs, action, reward, discount, next_obs, label, step, ids, meta = Utils.to_torch(
            batch, self.device)

        # "Envision" / "Perceive"

        # Augment
        obs = self.aug(obs)

        # Classification conversion  Can it be set in replay? Should replay have a to(device) method, no to_torch here
        # But then envs should probably convert to numpy/cpu too, not in env? Collate speicifc device? DataParallel
        # if self.classify:
        #     label = label.long()

        # RL -> Imitation Learning conversion
        # if self.imitate:
        #     label = Utils.one_hot(action, self.num_actions) if self.discrete and not self.classify \
        #         else action if self.discrete or not self.classify \
        #         else action.argmax(-1)

        # Actor-Critic -> Generator-Discriminator conversion
        if self.generate:
            obs = (obs - self.low) * 2 / (self.high - self.low) - 1  # Normalize first
            action, reward[:] = obs.flatten(-3), 1
            next_obs[:] = label[:] = float('nan')

        # Encode
        obs = self.encoder(obs)

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
            Pi = self.actor(obs[instruction])

            y_predicted = (Pi.All_Qs if self.discrete else Pi.mean).mean(1)

            # Inference
            # y_predicted = self.actor(obs[instruction], self.step).mean[:, 0]

            mistake = cross_entropy(y_predicted, label[instruction].long(), reduction='none')
            # mistake = cross_entropy if self.classify else mse(y_predicted, label[instruction].long(), reduction='none')
            # if self.classify:
            correct = (torch.argmax(y_predicted, -1) == label[instruction]).float()
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
                half = len(instruction) // 2
                mistake[:half] = cross_entropy(y_predicted[:half].uniform_(-1, 1),
                                               label[instruction][:half].long(), reduction='none')
                action[instruction] = y_predicted.detach()
                reward[instruction] = -mistake[:, None].detach()  # reward = -error
                next_obs[instruction] = float('nan')

                if self.log:
                    logs.update({'reward': reward})

        # Reinforcement learning / generative modeling
        if self.RL or self.generate:
            # "Imagine"

            # Generative modeling
            if self.generate:
                half = len(obs) // 2

                Pi = self.actor(obs[:half], self.step)
                generated_image = Pi.best if self.discrete else Pi.mean
                # generated_image = self.actor(obs[:half], self.step).mean[:, 0]  # TODO Don't collapse n'-dim in Pi

                action[:half], reward[:half] = generated_image[:, 0], 0  # Discriminate

            # "Discern"

            # Critic loss
            critic_loss = QLearning.ensembleQLearning(self.critic, self.actor,
                                                      obs, action, reward, discount, next_obs,
                                                      self.step, self.num_actions, logs=logs)

            # Update critic
            Utils.optimize(critic_loss,
                           self.critic, epoch=self.epoch if offline else self.episode)

        # Update encoder
        if not self.generate:
            Utils.optimize(None,  # Using gradients from previous losses
                           self.encoder, epoch=self.epoch if offline else self.episode)

        if self.generate or self.RL and not self.discrete:
            # "Change" / "Grow"

            # Actor loss
            actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                           self.step, self.num_actions, logs=logs)

            # Update actor
            Utils.optimize(actor_loss,
                           self.actor, epoch=self.epoch if offline else self.episode)

        return logs
