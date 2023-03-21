# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch.nn.functional import mse_loss, binary_cross_entropy


def ensembleQLearning(critic, actor, obs, action, reward, discount=1, next_obs=None, step=0, logs=None):
    # Non-empty next_obs
    has_future = next_obs is not None and bool(next_obs.nelement())

    # Compute Bellman target
    with torch.no_grad():
        # Current reward
        target_Q = reward

        # Future action and Q-values
        next_action = All_Next_Qs = None

        # Discounted future reward
        if has_future:
            # Get actions for next_obs
            next_Pi = actor(next_obs, step)

            # Discrete Critic tabulates all actions for single-dim discrete envs a priori, no need to sample
            if not critic.all_actions_known:
                next_action = next_Pi.sample(1)  # Sample

            if actor.discrete:
                All_Next_Qs = next_Pi.All_Qs  # Discrete Actor policy already knows all Q-values

            # Q-values per action
            next_Qs = critic.ema.eval()(next_obs, next_action, All_Next_Qs)  # Call a delayed-copy of Critic: Q(obs, a)

            # Pessimistic Q-values per action
            next_q = next_Qs.min(1)[0]  # Min-reduced critic ensemble

            # Weigh each action's pessimistic Q-value by its probability
            next_action_prob = next_q.size(1) < 2 or next_Pi.log_prob(next_action).softmax(-1)  # Action probability
            next_v = (next_q * next_action_prob).sum(-1, keepdim=True)  # Expected Q-value = E_a[Q(obs, a)]

            target_Q += discount * next_v  # Add expected future discounted-cumulative-reward to reward

    Qs = critic(obs, action)  # Q-ensemble

    # Use BCE if Critic ends with Sigmoid, otherwise MSE
    criterion = binary_cross_entropy if critic.binary else mse_loss

    # Temporal difference (TD) error
    q_loss = criterion(Qs, target_Q.unsqueeze(1).expand_as(Qs))

    if logs is not None:
        logs['temporal_difference_error'] = q_loss
        logs.update({f'q{i}': Qs[:, i].to('cpu' if Qs.device.type == 'mps' else Qs).median()  # torch.median not on mps
                     for i in range(Qs.shape[1])})
        logs['target_q'] = target_Q.mean()

    return q_loss
