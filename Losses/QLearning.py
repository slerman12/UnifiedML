# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
import torch.nn.functional as F


def ensembleQLearning(critic, actor, obs, action, reward, discount, next_obs, step,
                      num_actions=1, one_hot=False, one_hot_next=False, logs=None):
    # Non-NaN next_obs
    has_future = ~torch.isnan(next_obs.flatten(1)[:, :1]).squeeze(1) * bool(next_obs.size(1))
    next_obs = next_obs[has_future]

    # One-hot encoding in case discrete actions need to be treated as continuous or vice versa
    # if one_hot:
    #     action = Utils.one_hot(action, critic.action_dim) * 2 - 1 if action.shape[-1] == 1 \
    #         else Utils.rone_hot(action, null_value=-1)

    # Compute Bellman target
    with torch.no_grad():
        # Current reward
        target_Q = reward

        # Future action and Q-values
        next_action = All_Next_Qs = None

        # Discounted future reward
        if has_future.any():
            # Get actions for next_obs
            next_Pi = actor(next_obs, step)

            # Discrete Critic knows all actions for discrete Envs a priori, no need to sample
            all_actions_known = hasattr(critic, 'action')

            if not all_actions_known:
                next_action = next_Pi.rsample(num_actions)  # Sample actions

            if actor.discrete:
                All_Next_Qs = next_Pi.All_Qs  # Discrete Actor policy already knows all Q-values

            # if critic.discrete:
            #     next_action, next_action_log_probs = None, 0  # Discrete critic uses all actions, no need to sample
            # else:
            #     next_Pi = actor(next_obs, step)  # Sampling actions
            #
            #     # One-hot or sample
            #     next_action = torch.eye(critic.action_dim,
            #                             device=obs.device).expand(next_obs.shape[0], -1, -1) * 2 - 1 if one_hot_next \
            #         else next_Pi.rsample(num_actions)
            #     next_action_log_probs = next_Pi.log_prob(next_action).sum(-1, keepdim=True).flatten(1)

            # Q-values per action
            next_Qs = critic.ema(next_obs, next_action, All_Next_Qs)
            next_q = next_Qs.min(1)[0]  # Min-reduced ensemble

            # Weigh each action's Q-value by its probability
            next_v = torch.zeros_like(discount)
            next_probs = torch.softmax(next_q - next_q.max(-1, keepdim=True)[0], -1)
            next_v[has_future] = torch.sum(next_q * next_probs, -1, keepdim=True)

            target_Q += discount * next_v

    Qs = critic(obs, action)  # Q-ensemble

    # Temporal difference (TD) error (via MSE, but could also use Huber)
    q_loss = F.mse_loss(Qs, target_Q.unsqueeze(1).expand_as(Qs))

    # REMOVED
    # Re-prioritize based on certainty e.g., https://arxiv.org/pdf/2007.04938.pdf
    # q_loss = td_error * torch.sigmoid(-Q.stddev * priority_temp) + 0.5

    # if reduction == 'mean':
    #     q_loss = q_loss.mean()

    if logs is not None:
        logs['temporal_difference_error'] = q_loss
        logs.update({f'q{i}': Qs[:, i].median() for i in range(Qs.shape[1])})
        logs['target_q'] = target_Q.mean()
        # logs['q_loss'] = q_loss.mean()

    return q_loss
