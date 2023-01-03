# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x, generator=None):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype,
                              generator=generator)

        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, norm_tanh, ln_tanh):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = [nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True)]
        if norm_tanh:
            self.policy += [utils.EmbedNorm()]
        if ln_tanh:
            self.policy += [nn.LayerNorm(hidden_dim, elementwise_affine=False)]
        self.policy += [nn.Linear(hidden_dim, action_shape[0])]
        self.policy = nn.Sequential(*self.policy)


        self.apply(utils.weight_init)

        self.share_trunk = False
        self.last_tanh = None



    def forward(self, obs, std, return_pretanh=False):
        h = self.trunk(obs)
        if self.share_trunk:
            h = h.detach()
        mu = self.policy(h)
        pretanh = mu
        mu = torch.tanh(mu)
        self.last_tanh = mu.detach()
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return (dist, pretanh) if return_pretanh else dist



class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim, qnorm, ln_critic):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())


        self.Q1 = [nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)] 
        if qnorm:
            self.Q1 += [utils.EmbedNorm()]
        if ln_critic:
             self.Q1 += [nn.LayerNorm(hidden_dim, elementwise_affine=False)]

        self.Q1 += [nn.Linear(hidden_dim, 1)]
        self.Q1 = nn.Sequential(*self.Q1)

        self.Q2 = [nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)] 
        if qnorm:
            self.Q2 += [utils.EmbedNorm()]
        if ln_critic:
             self.Q2 += [nn.LayerNorm(hidden_dim, elementwise_affine=False)]
        self.Q2 += [nn.Linear(hidden_dim, 1)]
        self.Q2 = nn.Sequential(*self.Q2)


        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, 
                 use_tb, qnorm, freeze_encoder, q_reduction, freeze_trunks,
                 warmup_encoder, cpc_until, share_trunk, share_generator, 
                 log_vector_stats, norm_tanh, ln_tanh, logabsdet_penalty, square_penalty, penalty_schedule, actor_lr, linear_lr_scheduler, loss_func, ln_critic, grad_clip):


        assert not (norm_tanh and ln_tanh), 'chose one'
        assert logabsdet_penalty is None or square_penalty is None, 'at most one'
        assert penalty_schedule is None or square_penalty is not None, 'schedule require value'
        assert linear_lr_scheduler is None or warmup_encoder is None, 'at most one schedule'


        actor_lr = lr if actor_lr is None else actor_lr

        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.cpc_until = cpc_until
        self.log_vector_stats = log_vector_stats
        self.logabsdet_penalty = logabsdet_penalty
        self.square_penalty = square_penalty
        self.penalty_schedule = 'linear({},0.0,500000)'.format(self.square_penalty) if penalty_schedule is not None else None
        self.linear_lr_scheduler = linear_lr_scheduler
        self.grad_clip = grad_clip

        if loss_func == 'mse':
            self.q_loss_func = F.mse_loss 
        elif loss_func == 'quartic':
            self.q_loss_func = utils.quartic_loss
        else:
            raise Exception('uknown')
        print('self.q_loss_func', self.q_loss_func)

        if share_generator:
            self.obs_gen = torch.Generator(device=device)
            self.next_obs_gen = torch.Generator(device=device)
            self.obs_gen.set_state(self.next_obs_gen.get_state())
            print('using shared generators')
        else:
            self.obs_gen, self.next_obs_gen = None, None



        if q_reduction == 'mean':
            self.reduction = utils.avg
        elif q_reduction == 'min':
            self.reduction = torch.min
        else:
            raise Exception('unknown reduction')


        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim, norm_tanh, ln_tanh).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim, qnorm, ln_critic).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim, qnorm, ln_critic).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        if cpc_until is not None:
            self.cpc_predictor = utils.CPCPredictor(feature_dim, feature_dim, hidden_size=512).to(device)
            self.ema_encoder = Encoder(obs_shape).to(device)
            self.ema_encoder.load_state_dict(self.encoder.state_dict())
            self.critic_opt = torch.optim.Adam(itertools.chain(self.critic.parameters(), self.cpc_predictor.parameters()), lr=lr)
        else:
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)


        assert not (freeze_trunks and cpc_until is not None), 'cpc requires trainable trunks'

        print(self.critic)
        print(self.actor)

        if freeze_trunks:
            self.actor_opt = torch.optim.Adam(self.actor.policy.parameters(), lr=actor_lr)
            self.critic_opt = torch.optim.Adam(itertools.chain(
						self.critic.Q1.parameters(),
						self.critic.Q2.parameters())
						, lr=lr)
        else:
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)

        self.warmup_encoder = warmup_encoder
        if warmup_encoder is not None:
            self.encoder_scheduler = utils.get_warmup_scheduler(self.encoder_opt, warmup_encoder)
            self.actor_scheduler = utils.get_warmup_scheduler(self.actor_opt, warmup_encoder)
            self.critic_scheduler = utils.get_warmup_scheduler(self.critic_opt, warmup_encoder)
        elif linear_lr_scheduler is not None:
            self.encoder_scheduler = utils.get_linear_scheduler(self.encoder_opt, linear_lr_scheduler, 0.1)
            self.critic_scheduler = utils.get_linear_scheduler(self.critic_opt, linear_lr_scheduler, 0.1)

        # maybe freeze
        if freeze_encoder:
            self.encoder_opt = utils.NoUpdateOptWrapper(self.encoder_opt)


        if share_trunk:
            self.actor.trunk = self.critic.trunk
            self.actor.share_trunk = True

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]


    def update_critic(self, obs, action, reward, discount, next_obs, step, next_obs_orig):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = self.reduction(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = self.q_loss_func(Q1, target_Q) + self.q_loss_func(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()


        if self.cpc_until is not None and step < self.cpc_until:
            cpc_loss = self.get_cpc_loss(obs, next_obs_orig)
            critic_loss = critic_loss + cpc_loss

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()

        # maybe clip grads
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)

        self.critic_opt.step()
        self.encoder_opt.step()

        if self.warmup_encoder is not None or self.linear_lr_scheduler is not None:
            self.critic_scheduler.step()
            self.encoder_scheduler.step()
            if self.use_tb:
                crit_lr = self.critic_scheduler.get_last_lr()
                metrics['critic_lr'] = crit_lr[0]


        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist, pretanh = self.actor(obs, stddev, return_pretanh=True)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = self.reduction(Q1, Q2)
        actor_loss = -Q.mean()

        if self.logabsdet_penalty is not None:
            penalty = utils.log_abs_det(pretanh).mean()
            actor_loss = actor_loss - self.logabsdet_penalty * penalty


        if self.square_penalty is not None:
            penalty = pretanh.square().mean()
            actor_loss = actor_loss + self.square_penalty * penalty


        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()

        # maybe clip grads
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)

        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_abs_tanh'] = self.actor.last_tanh.abs().mean().item()
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            if self.logabsdet_penalty is not None:
                metrics['actor_penalty'] = penalty.item()
            if self.square_penalty is not None: 
                metrics['actor_penalty_square'] = penalty.item()


        if self.warmup_encoder is not None:
            self.actor_scheduler.step()

        return metrics


    def get_cpc_loss(self, obs, next_obs_orig):
        with torch.no_grad():
            obs_other = self.ema_encoder(next_obs_orig)
            obs_other = F.normalize(self.critic_target.trunk[0](obs_other))
        obs = F.normalize(self.cpc_predictor(self.critic.trunk[0](obs)))
        cpc_loss = - (obs*obs_other.detach())
        cpc_loss = cpc_loss.sum(dim=-1).mean()
        return cpc_loss


    def calculate_vector_stats(self):
        metrics = {}
        for name, net, opt in zip(['critic', 'actor', 'encoder'], [self.critic, self.actor, self.encoder], [self.critic_opt, self.actor_opt, self.encoder_opt]):
            metrics['{}_weight_norm'.format(name)] = torch.norm(utils.get_flat_weights(net)).item()
            metrics['{}_grad_norm'.format(name)] = torch.norm(utils.get_flat_grads(net)).item()
            metrics['{}_update_norm'.format(name)] = torch.norm(utils.get_flat_adam_update(opt)).item()
        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float(), generator=self.obs_gen)
        next_obs = self.aug(next_obs.float(), generator=self.next_obs_gen)
        next_obs_orig = next_obs
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step, next_obs_orig))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))


        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        if self.cpc_until is not None and step < self.cpc_until:
            utils.soft_update_params(self.encoder, self.ema_encoder,
                                 self.critic_target_tau)


        if self.log_vector_stats:
            metrics.update(self.calculate_vector_stats())


        if self.penalty_schedule is not None:
            self.square_penalty = utils.schedule(self.penalty_schedule, step)
            if self.use_tb:
                metrics['square_penalty_coeff'] = self.square_penalty

        return metrics
