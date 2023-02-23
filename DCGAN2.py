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

discriminator = Discriminator().to(device)
generator = Generator().to(device)

criterion = nn.BCELoss()

discriminator_optim = Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
generator_optim = Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))


for epoch in range(num_epochs):
    for i, (obs, *_) in enumerate(dataloader):

        rand = torch.randn((len(obs), z_dim, 1, 1), device=device)
        action_ = generator(rand)

        # Discriminate Real
        action = obs.view_as(action_).to(device)
        Qs = discriminator(action.detach())
        reward = torch.ones_like(Qs)
        target_Q = reward

        critic_loss = criterion(Qs, target_Q)

        # Discriminate Plausible
        reward = torch.zeros_like(Qs)

        Qs = discriminator(action_.detach())
        Q_target = reward

        critic_loss += criterion(Qs, Q_target)
        discriminator_optim.zero_grad()
        critic_loss.backward()
        discriminator_optim.step()

        # Generate
        Qs = discriminator(action_)
        Q_target = torch.ones_like(Qs)

        actor_loss = criterion(Qs, Q_target)
        generator_optim.zero_grad()
        actor_loss.backward()
        generator_optim.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d]' % (epoch, num_epochs, i, len(dataloader)))

plt.figure(figsize=(15, 15))

plt.subplot(1, 2, 1)
plt.axis('off')
plt.title('Real')
plt.imshow(np.transpose(vutils.make_grid(obs[:64].detach(), padding=5, normalize=True).cpu(), (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('Plausible Not-Real')
action = generator(rand).view_as(obs)
plt.imshow(np.transpose(vutils.make_grid(action[:64].detach(), padding=2, normalize=True).cpu(), (1, 2, 0)))

path = Path('./Benchmarking/DCGAN/AC2Agent/classify/CelebA_1_Video_Image')
path.mkdir(parents=True, exist_ok=True)
plt.savefig(path / 'generated.png')

plt.close()
