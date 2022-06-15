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
from Blocks.Actors import EnsembleGaussianActor, CategoricalCriticActor
from Blocks.Critics import EnsembleQCritic

from Learner import F_ckGradientDescent

from Losses import QLearning, PolicyLearning


class MetaDQNAgent(torch.nn.Module):
    """Deep Q Network
    Generalized to continuous action spaces, classification, and generative modeling
    Substitutes backwards pass with custom loss propagation and custom optimizer"""
    def __init__(self,
                 obs_shape, action_shape, trunk_dim, hidden_dim, data_stats, recipes,  # Architecture
                 lr, lr_decay_epochs, weight_decay, ema_decay, ema,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, supervise, generate, device, parallel, log,  # On-boarding
                 num_actions=1, num_critics=2, step_optim_per_learn=1  # MetaDQN
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
        self.action_dim = math.prod(obs_shape) if generate else action_shape[-1]
        self.meta_shape = (3,)

        self.num_actions = num_actions  # Num actions sampled by actor
        self.step_optim_per_learn = step_optim_per_learn  # How often to step optimizer
        self.learn_step = 0  # Count learn steps

        for name in ('encoder', 'actor', 'critic'):
            del getattr(recipes, name)['optim']

        self.encoder = Utils.Rand(trunk_dim) if generate \
            else CNNEncoder(obs_shape, data_stats=data_stats, **recipes.encoder, lr=lr, lr_decay_epochs=lr_decay_epochs,
                            weight_decay=weight_decay, ema_decay=ema_decay * ema, parallel=parallel,
                            optim=torch.optim.SGD)

        repr_shape = (trunk_dim,) if generate \
            else self.encoder.repr_shape

        # Continuous actions
        self.actor = None if self.discrete \
            else EnsembleGaussianActor(repr_shape, trunk_dim, hidden_dim, self.action_dim, **recipes.actor,
                                       ensemble_size=1, stddev_schedule=stddev_schedule, stddev_clip=stddev_clip,
                                       lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                       ema_decay=ema_decay * ema, optim=torch.optim.SGD)

        self.critic = EnsembleQCritic(repr_shape, trunk_dim, hidden_dim, self.action_dim, **recipes.critic,
                                      ensemble_size=num_critics, discrete=self.discrete, ignore_obs=generate,
                                      lr=lr, lr_decay_epochs=lr_decay_epochs, weight_decay=weight_decay,
                                      ema_decay=ema_decay, optim=torch.optim.SGD)

        self.action_selector = CategoricalCriticActor(stddev_schedule)

        # Image augmentation
        self.aug = Utils.instantiate(recipes.aug) or (IntensityAug(0.05) if discrete
                                                      else RandomShiftsAug(pad=4))

        # Custom optimizer
        for name in ('encoder', 'actor', 'critic'):

            block = getattr(self, name)

            setattr(block, 'optim',
                    F_ckGradientDescent(block.optim) if hasattr(block, 'optim')
                    else None)

        # Birth

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.encoder, self.actor, self.critic):
            obs = torch.as_tensor(obs, device=self.device)

            # EMA shadows
            encoder = self.encoder.ema if self.ema and not self.generate else self.encoder
            actor = self.actor.ema if self.ema and not self.discrete else self.actor
            critic = self.critic.ema if self.ema else self.critic

            # "See"
            obs = encoder(obs)

            actions = None if self.discrete \
                else actor(obs, self.step).sample(self.num_actions) if self.training \
                else actor(obs, self.step).mean

            # DQN action selector is based on critic
            Pi = self.action_selector(critic(obs, actions), self.step)

            action = Pi.sample() if self.training \
                else Pi.best

            if self.training:
                self.step += 1
                self.frame += len(obs)

                # Explore phase
                if self.step < self.explore_steps and not self.generate:
                    action = torch.randint(self.action_dim, size=action.shape) if self.discrete \
                        else action.uniform_(-1, 1)

            return action

    # "Dream"
    def learn(self, replay):
        with torch.no_grad():
            # "Recollect"

            batch = next(replay)
            obs, action, reward, discount, next_obs, label, *traj, step, ids, meta = Utils.to_torch(
                batch, self.device)

            # Actor-Critic -> Generator-Discriminator conversion
            if self.generate:
                action, reward[:] = obs.flatten(-3) / 127.5 - 1, 1
                next_obs[:] = label[:] = float('nan')

            batch_size = obs.shape[0]

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

            if replay.offline:
                self.step += 1
                self.frame += len(obs)
                self.epoch = replay.epoch

            # if self.epoch > 1:
            #     assert (meta[:, 0] == label + self.epoch).all()
                # if (meta[:, 0] == label + self.epoch).all():
                #     print('good')
                # else:
                #     print(meta[:, 0], '\n', label)
                #     time.sleep(2)

            self.learn_step += 1
            step_optim = self.learn_step % self.step_optim_per_learn == 0

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

                    # Forward-prop
                    self.encoder.optim.propagate(mistake, meta[:, 0], batch_size)
                    self.actor.optim.propagate(mistake, meta[:, 0], batch_size)

                    # meta[:, 0][meta[:, 0].isnan()] = mistake[meta[:, 0].isnan()]
                    # meta[:, 0] = torch.max(mistake, meta[:, 0])
                    meta[:, 0] = mistake

                    # Update supervised
                    Utils.optimize(None,
                                   self.actor, epoch=self.epoch if replay.offline else self.episode,
                                   step_optim=step_optim)

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
                                                          obs, action, reward, discount, next_obs,
                                                          self.step, self.num_actions, logs=logs, reduction='none')

                # Forward-prop todo retain graph above
                self.encoder.optim.propagate(critic_loss, meta[:, 1], batch_size)
                self.critic.optim.propagate(critic_loss, meta[:, 1], batch_size)

                # meta[:, 1][meta[:, 1].isnan()] = critic_loss[meta[:, 1].isnan()]
                # meta[:, 1] = torch.min(critic_loss, meta[:, 1])
                meta[:, 1] = critic_loss

                # Update critic
                Utils.optimize(None,
                               self.critic, epoch=self.epoch if replay.offline else self.episode,
                               step_optim=step_optim)

            # Update encoder
            if not self.generate:
                Utils.optimize(None,
                               self.encoder, epoch=self.epoch if replay.offline else self.episode,
                               step_optim=step_optim)

            if self.generate or self.RL and not self.discrete:
                # "Change" / "Grow"

                # Actor loss
                actor_loss = PolicyLearning.deepPolicyGradient(self.actor, self.critic, obs.detach(),
                                                               self.step, self.num_actions, logs=logs, reduction='none')

                self.actor.optim.propagate(actor_loss, meta[:, 2], batch_size)

                # meta[:, 2][meta[:, 2].isnan()] = actor_loss[meta[:, 2].isnan()]
                # meta[:, 2] = torch.min(actor_loss, meta[:, 2])
                meta[:, 2] = actor_loss

                # Update actor
                Utils.optimize(None,
                               self.actor, epoch=self.epoch if replay.offline else self.episode, backward=False,
                               step_optim=step_optim)

            # meta[:, 0] = label + self.epoch + 1
            replay.rewrite({'meta': meta}, ids)
            # threading.Thread(target=replay.rewrite, args=({'meta': meta}, ids)).start()

            return logs
