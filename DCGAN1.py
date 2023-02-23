# Template created by Sam Lerman, slerman@ur.rochester.edu.

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

import torchvision.transforms as transforms
import torchvision.utils as vutils
from CelebA import CelebA
# from torchvision.datasets.celeba import CelebA

from Discriminator import Discriminator
from Generator import Generator


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

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

discriminator = Discriminator()
generator = Generator()

criterion = nn.BCELoss()

discriminator_optim = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
generator_optim = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))


for epoch in range(num_epochs):
    for i, (obs, *_) in enumerate(dataloader):

        discriminator_optim.zero_grad()
        generator_optim.zero_grad()

        obs = obs.to(device)

        # Train Discriminator
        rand = torch.randn((obs[0].shape[0], z_dim), device=obs[0].device)
        action_ = generator(rand)
        action = torch.cat([obs.view_as(action_), action_], 0)
        reward_ = torch.zeros((len(obs), 1)).to(obs)
        reward = torch.cat([torch.ones_like(reward_), reward_], 0)

        Qs = discriminator(action.detach())
        target_Q = reward

        critic_loss = criterion(Qs, target_Q)
        critic_loss.backward()
        discriminator_optim.step()

        # Train Generator
        Qs = discriminator(action_)
        Q_target = torch.ones_like(Qs)
        actor_loss = criterion(Qs, Q_target)
        actor_loss.backward()
        generator_optim.step()

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
action = generator(obs.to(device)).view_as(obs)
plt.imshow(np.transpose(vutils.make_grid(action[:64].detach(), padding=2, normalize=True).cpu(), (1, 2, 0)))

path = Path('./Benchmarking/DCGAN/AC2Agent/classify/CelebA_1_Video_Image')
path.mkdir(parents=True, exist_ok=True)
plt.savefig(path / 'generated.png')

plt.close()
