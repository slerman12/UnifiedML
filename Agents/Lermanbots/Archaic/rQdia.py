# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

import Utils

from Agents import DrQV2Agent

from Losses.PolicyLearning import deepPolicyGradient
from Losses.QLearning import ensembleQLearning
from Losses.InvarianceLearning import rQdia


class rQdiaAgent(DrQV2Agent):
    # scaler = Utils.Scaler(num_chunks=1)  # divides batches up into computationally-affordable chunks

    # @Utils.loop_in_chunks(scaler=scaler)  # repeats the update for each chunk
    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter) \
        #     if self.scaler.cached_batch is None else self.scaler.cached_batch
        # batch = self.scaler(batch)

        obs, action, reward, discount, next_obs, *traj = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r = traj

        # for rQdia
        obs_orig = self.encoder(obs)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())

        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # clear grads
        # if self.scaler.first_iteration:
        self.encoder.optim.zero_grad(set_to_none=True)
        self.critic.optim.zero_grad(set_to_none=True)
        self.actor.optim.zero_grad(set_to_none=True)

        # critic loss
        critic_loss = ensembleQLearning(self.actor, self.critic,
                                        obs, action, reward, discount, next_obs, step,
                                        logs=metrics if self.use_tb else None)

        # rQdia
        critic_loss += rQdia(self.critic, obs, obs_orig, action, n_scaling=0.4, m_scaling=0.4)

        # critic_loss /= self.scaler.num_chunks

        # update critic, encoder
        critic_loss.backward()
        # if self.scaler.last_iteration:  # accumulate gradients until last chunk - seems to not work...?
        self.critic.optim.step()
        self.encoder.optim.step()

        if not self.discrete:
            # actor loss
            actor_loss = deepPolicyGradient(self.actor, self.critic,
                                            obs.detach(), step,
                                            logs=metrics if self.use_tb else None)

            # actor_loss /= self.scaler.num_chunks

            # update actor
            actor_loss.backward()
            # if self.scaler.last_iteration:
            self.actor.optim.step()

        # update critic target
        self.critic.update_target()

        return metrics
