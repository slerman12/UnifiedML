import numpy as np
import torch

import Utils

from Agents import DQNDPGAgent

from Blocks.Actors import DiagonalGaussianActor, CategoricalCriticActor

from Losses.PolicyLearning import deepPolicyGradient, entropyMaxim
from Losses.QLearning import ensembleQLearning


class SACAgent(DQNDPGAgent):
    """SAC algorithm."""
    def __init__(self,
                 obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
                 lr, update_per_steps, target_tau,  # Optimization
                 explore_steps, init_temperature, log_std_bounds,   # Exploration
                 discrete, device, log_tensorboard  # On-boarding
                 ):
        super().__init__(
            obs_shape, action_shape, feature_dim, hidden_dim,  # Architecture
            lr, update_per_steps, target_tau,  # Optimization
            explore_steps, 0, 0,  # Exploration
            discrete, device, log_tensorboard  # On-boarding
        )

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)  # init_temperature=0.1
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_shape[-1]  # todo even for discrete?

        self.log_alpha.__dict__['optim'] = torch.optim.Adam([self.log_alpha], lr=lr)  # todo is list needed?

        # is this actor necessary (as opposed to the TruncatedGaussian)? todo categorical temp...
        self.actor = CategoricalCriticActor(self.critic, temp=self.alpha) if discrete \
            else DiagonalGaussianActor(self.encoder.repr_dim, feature_dim, hidden_dim,
                                       action_shape[-1], log_std_bounds).to(device)  # log_std_bounds=[-5, 2]

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @Utils.optimize('encoder', 'critic')
    # Critic loss
    def update_critic(self, obs, action, reward, discount, next_obs, dist=None, logs=None):
        return ensembleQLearning(self.actor, self.critic, obs, action, reward, discount, next_obs, self.step,
                                 dist=dist, entropy_temp=self.alpha,  # todo put 'entropy_temp' in super / delete ?
                                 logs=logs if self.use_tensorboard else None)

    @Utils.optimize('actor')
    # Actor loss
    def update_actor(self, obs, dist=None, logs=None):
        # Entropy loss
        actor_loss = entropyMaxim(self.actor, obs.detach(), self.step, self.alpha, dist,
                                  logs=logs if self.use_tensorboard else None)

        if not self.discrete:
            # Policy loss
            actor_loss += deepPolicyGradient(self.actor, self.critic, obs.detach(), self.step, dist,
                                             logs=logs if self.use_tensorboard else None)

        return actor_loss

    @Utils.optimize('log_alpha')
    # Entropy temp loss
    def update_misc(self, obs, action, reward, discount, next_obs, traj_o, traj_a, traj_r, dist, logs=None):
        super().update_misc(obs, action, reward, discount, next_obs, traj_o, traj_a, traj_r, dist, logs)

        log_pi = dist.log_prob(action).sum(-1, keepdim=True)
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if self.log_tensorboard:
            logs['alpha_loss'] = alpha_loss.item()
            logs['alpha_value'] = self.alpha

        return alpha_loss
