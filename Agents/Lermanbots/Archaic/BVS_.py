# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

import Utils

from Agents import DrQV2Agent

from Blocks.Encoders import LayerNormMLPEncoder
from Blocks.Critics import EnsembleQCritic
from Blocks.Creators import SubPlanner

from Losses.QLearning import ensembleQLearning
from Losses.PolicyLearning import deepPolicyGradient
from Losses.SelfSupervisedLearning import bootstrapLearningBVS


class BVSAgent(DrQV2Agent):
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # architecture
                 target_tau, stddev_schedule, stddev_clip, plan_discount,  # models
                 lr, update_every_steps,  # optimization
                 num_expl_steps,  # exploration
                 discrete, device, use_tb  # on-boarding
                 ):
        super().__init__(
            obs_shape, action_shape, feature_dim, hidden_dim,  # architecture
            stddev_schedule, stddev_clip, target_tau,  # models
            lr, update_every_steps,  # optimization
            num_expl_steps,  # exploration
            discrete, device, use_tb  # on-boarding
        )

        self.plan_discount = plan_discount

        # models
        # state based
        # self.sub_planner = LayerNormMLPEncoder(self.encoder.repr_dim, feature_dim, hidden_dim, hidden_dim,
        #                                        target_tau=target_tau, optim_lr=lr).to(device)
        # state-action based
        self.sub_planner = SubPlanner(self.encoder.repr_dim, feature_dim, hidden_dim, hidden_dim, action_shape[-1],
                                      target_tau=target_tau, optim_lr=lr, discrete=discrete).to(device)

        self.planner = LayerNormMLPEncoder(hidden_dim, hidden_dim, hidden_dim, hidden_dim,
                                           target_tau=target_tau, optim_lr=lr).to(device)

        self.critic = EnsembleQCritic(hidden_dim, hidden_dim, hidden_dim, action_shape[-1],
                                      target_tau=target_tau, optim_lr=lr, discrete=discrete).to(device)

    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, *traj = Utils.to_torch(
            batch, self.device)
        traj_o, traj_a, traj_r = traj

        # augment
        traj_o = traj_o.float()
        for i in range(traj_o.shape[1]):
            traj_o[:, i] = self.aug(traj_o[:, i])

        # encode
        obs = self.encoder(traj_o[:, 0])
        with torch.no_grad():
            traj_o = self.encoder(traj_o)
            next_obs = traj_o[:, -1]

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # clear grads
        self.critic.optim.zero_grad(set_to_none=True)
        self.planner.optim.zero_grad(set_to_none=True)
        self.sub_planner.optim.zero_grad(set_to_none=True)
        self.encoder.optim.zero_grad(set_to_none=True)

        # critic loss
        critic_loss = ensembleQLearning(self.actor, self.critic,
                                        obs, action, reward, discount, next_obs, step,
                                        sub_planner=self.sub_planner, planner=self.planner,
                                        logs=metrics if self.use_tb else None)

        # update critic, planners, encoder
        critic_loss.backward()
        self.critic.optim.step()
        self.encoder.optim.step()

        # planner loss
        planner_loss = bootstrapLearningBVS(self.actor, self.sub_planner, self.planner,
                                            obs, traj_o, self.plan_discount,
                                            traj_a, step,  # comment out for state-based
                                            logs=metrics if self.use_tb else None)

        # update planner
        planner_loss.backward()
        self.planner.optim.step()
        self.sub_planner.optim.step()

        if not self.discrete:
            # clear grads
            self.actor.optim.zero_grad(set_to_none=True)

            # actor loss
            actor_loss = deepPolicyGradient(self.actor, self.critic,
                                            obs.detach(), step, sub_planner=self.sub_planner, planner=self.planner,
                                            logs=metrics if self.use_tb else None)

            # update actor
            actor_loss.backward()
            self.actor.optim.step()

        # update critic target
        self.critic.update_target_params()

        # update planner targets
        self.sub_planner.update_target()
        self.planner.update_target()

        return metrics
