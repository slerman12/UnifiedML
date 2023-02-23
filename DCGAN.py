from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np

from Blocks.Architectures.Vision.DCGAN import Generator, Discriminator

from Blocks.Encoders import CNNEncoder
from Blocks.Actors import EnsemblePiActor
from Blocks.Critics import EnsembleQCritic

from Datasets.Suites._CelebA import CelebA
from Datasets.Suites.Classify import AttrDict

from Losses import QLearning

import Utils

import matplotlib.pyplot as plt


Utils.set_seeds(0)

batch_size = 256
num_epochs = 5
z_dim = 100
lr = 0.0002
beta1 = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = CelebA(root="Datasets/ReplayBuffer/Classify/CelebA_Train/",
                 download=True,
                 transform=transforms.Compose([
                     transforms.Resize(64),
                     transforms.CenterCrop(64),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Encoder can standardize
                 ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

obs_spec = AttrDict({'shape': [3, 64, 64], 'mean': 0.5, 'stddev': 0.5, 'low': 0, 'high': 1})
action_spec = AttrDict({'shape': obs_spec.shape, 'discrete_bins': None, 'low': -1, 'high': 1, 'discrete': False})

encoder = CNNEncoder(obs_spec, standardize=False, Eyes=nn.Identity)
actor = EnsemblePiActor(encoder.repr_shape, 100, -1, action_spec, trunk=Utils.Rand, Pi_head=Generator, ensemble_size=1,
                        lr=lr, optim={'_target_': 'Adam', 'betas': [beta1, 0.999]}).to(device)
critic = EnsembleQCritic(encoder.repr_shape, 100, -1, action_spec, Q_head=Discriminator, ensemble_size=1,
                         ignore_obs=True, lr=lr, optim={'_target_': 'Adam', 'betas': [beta1, 0.999]}).to(device)


optimizerG = optim.Adam(actor.parameters(), lr=lr, betas=(beta1, 0.999))
criterion = nn.BCELoss()


for epoch in range(num_epochs):
    for i, (obs, *_) in enumerate(dataloader):

        # Discriminate real
        obs = obs.to(device)
        obs = encoder(obs)
        action_ = actor(obs).mean  # Redundant, Slow
        action = obs.view_as(action_)
        reward = torch.ones((len(obs), 1)).to(obs)

        # Discriminate plausible
        half = len(action) // 2
        action[:half] = action_[:half]
        reward[:half] = 0
        critic_loss = QLearning.ensembleQLearning(critic, actor, obs, action, reward, 1, torch.ones(0), 1)

        Utils.optimize(critic_loss, critic)

        # Generate
        action = actor(obs).mean.view_as(obs)
        Qs = critic(obs, action).view(-1, 1)
        Q_target = torch.ones_like(reward)
        actor_loss = criterion(Qs[:half], Q_target[:half])

        Utils.optimize(actor_loss, actor)

        if i % 50 == 0:
            print('[%d/%d][%d/%d]' % (epoch, num_epochs, i, len(dataloader)))


obs, *_ = next(iter(dataloader))

plt.figure(figsize=(15, 15))

plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Real')
plt.imshow(np.transpose(vutils.make_grid(obs[:64].detach(), padding=5, normalize=True).cpu(), (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('Plausible Not-Real')
action = actor(obs.to(device)).mean.view_as(obs)
plt.imshow(np.transpose(vutils.make_grid(action[:64].detach(), padding=2, normalize=True).cpu(), (1, 2, 0)))

path = Path('./Benchmarking/DCGAN/AC2Agent/classify/CelebA_1_Video_Image')
path.mkdir(parents=True, exist_ok=True)
plt.savefig(path / 'generated.png')

plt.close()
