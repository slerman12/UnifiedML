# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import time

import torch

import Utils


class TwoHemispheresAgent(torch.nn.Module):
    """Two-Hemispheres Agent
    In order to avoid delays between inference and training"""
    def __init__(self,
                 obs_spec, action_spec, num_actions, trunk_dim, hidden_dim, standardize, norm, recipes,  # Architecture
                 lr, lr_decay_epochs, weight_decay, ema_decay, ema,  # Optimization
                 explore_steps, stddev_schedule, stddev_clip,  # Exploration
                 discrete, RL, supervise, generate, device, parallel, log,  # On-boarding
                 train_steps=1, Agent='Agents.AC2Agent'  # Two-Hemispheres Agent
                 ):
        super().__init__()

        self.device = device
        self.birthday = time.time()
        self.step = self.frame = 0
        self.episode = self.epoch = 1

        # Verify at least 2 GPUs
        assert True

        # Initialize agents, belonging to different GPUS

        # Birth

    def act(self, obs):
        with torch.no_grad(), Utils.act_mode(self.actor):
            obs = torch.as_tensor(obs, device=self.device).float()

            # Identity of self.Agent alternates
            action = self.Agent.act(obs)

            if self.training:
                self.step += 1
                self.frame += len(obs)

            return action

    # "Dream"
    def learn(self, replay=None):
        # Concurrently, separate CPU, if not already deployed:
        # 1. Load
        # 2. Train for self.train_steps
        # 3. Save
        # When done, switch agents

        return
