# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.


def deepPolicyGradient(actor, critic, obs, step, num_actions=1, reward=0, discount=1,
                       one_hot=False, logs=None):
    Pi = actor(obs, step)

    action = Pi.rsample(num_actions)
    # if one_hot:
    #     action = Utils.rone_hot(action, null_value=-1)

    Q = critic(obs, action)
    q, _ = Q.mean.min(1)

    q = reward + q * discount

    # REMOVED
    # Re-prioritize based on certainty e.g., https://arxiv.org/pdf/2007.04938.pdf
    # q *= torch.sigmoid(-Q.stddev * priority_temp) + 0.5

    policy_loss = -q.mean()

    # if reduction == 'mean':
    #     policy_loss = -q.mean()

    if logs is not None:
        logs['policy_loss'] = policy_loss
        logs['policy_prob'] = Pi.log_prob(action).exp().mean()
        logs['policy_q_stddev'] = Q.stddev.mean()

    return policy_loss
