# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch.nn.functional import binary_cross_entropy


def deepPolicyGradient(actor, critic, obs, step, num_actions=1, logs=None):
    Pi = actor(obs, step)

    action = Pi.rsample(num_actions)  # Differentiable sample via "re-parameterization"

    Qs = critic(obs, action)
    q, _ = Qs.min(1)  # Min-reduced ensemble

    if critic.binary:
        q = -binary_cross_entropy(q, torch.ones_like(q))
        # q = q.log()  # For numerical stability of maximizing Sigmoids
    #     can instead create ones and put on gpu via ones.type_as(q) and use -F.binary_cross_entropy(q, ones)

    policy_loss = -q.mean()  # Policy gradient ascent

    if logs is not None:
        logs['policy_loss'] = policy_loss
        logs['policy_prob'] = Pi.log_prob(action).exp().mean()

    return policy_loss
