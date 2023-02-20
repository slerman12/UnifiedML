from __future__ import print_function
import random
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from Datasets.Suites._CelebA import CelebA
from Blocks.Architectures.Vision.CNN import cnn_broadcast

from Blocks.Encoders import CNNEncoder
from Blocks.Actors import EnsemblePiActor
from Blocks.Critics import EnsembleQCritic
from Datasets.Suites.Classify import AttrDict

from Losses import QLearning

import Utils


seed = 999
torch.manual_seed(seed)
random.seed(seed)


dataroot = "Datasets/ReplayBuffer/Classify/CelebA_Train/"

batch_size = 128
image_size = 64
channels = 3

# Size of z latent vector (i.e. size of generator input)
z_dim = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

num_epochs = 5
lr = 0.0002
beta1 = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TODO Instead of normalize, standardize in Encoder and set those norm metrics to obs_spec
dataset = CelebA(root=dataroot,
                 download=True,
                 transform=transforms.Compose([
                     transforms.Resize(image_size),
                     transforms.CenterCrop(image_size),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 ]))


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


# Initializes model weights a la normal
def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)) or isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, input_shape, hidden_dim=64, output_shape=None):
        super().__init__()

        self.input_shape, self.output_shape = Utils.to_tuple(input_shape), Utils.to_tuple(output_shape)
        # Proprioceptive is channel dim
        self.input_shape = tuple(self.input_shape) + (1,) * (3 - len(self.input_shape))  # Broadcast input to 2D

        in_channels = self.input_shape[0]
        out_channels = in_channels if self.output_shape is None else self.output_shape[0]

        self.Generator = nn.Sequential(
            # (hidden_dim * 8) x 4 x 4
            nn.ConvTranspose2d(in_channels, hidden_dim * 8, 4, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True),

            # (hidden_dim * 4) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),

            # (hidden_dim * 2) x 16 x 16
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),

            # hidden_dim x 32 x 32
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),

            # out_channels x 64 x 64
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, 2, 1, bias=False),
            # nn.Identity() if self.output_shape is None else nn.AdaptiveAvgPool2d(self.output_shape[1:])  # Adapts scale
            nn.Tanh()  # TODO added here
        )

        self.apply(weight_init)

    def repr_shape(self, *_):
        return Utils.repr_shape(_, self.Generator)  # cnn_feature_shape doesn't support pre-computing ConvTranspose2d

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = cnn_broadcast(self.input_shape, x)

        x = self.Generator(x)

        # Restore leading dims
        out = x.view(*lead_shape, *(self.output_shape or x.shape[1:]))
        return out


# TODO uncommnet the adaptive pools
class Discriminator(nn.Module):
    def __init__(self, input_shape, hidden_dim=64, output_shape=None):
        super().__init__()

        self.input_shape, self.output_shape = Utils.to_tuple(input_shape), Utils.to_tuple(output_shape)

        in_channels = self.input_shape[0]

        self.Discriminator = nn.Sequential(
            # hidden_dim x 32 x 32
            # nn.AdaptiveAvgPool2d(64),  # Adapts from different scales  TODO Does averaging cause weird distortion?
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (hidden_dim * 2) x 16 x 16
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (hidden_dim * 4) x 8 x 8
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (hidden_dim * 8) x 4 x 4
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 1 x 1 x 1
            nn.Conv2d(hidden_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # TODO Using Sigmoid with MSE breaks for some reason!
        )

        self.apply(weight_init)

    def repr_shape(self, *_):
        return Utils.cnn_feature_shape(_, self.Discriminator)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = cnn_broadcast(self.input_shape, x)

        x = self.Discriminator(x)

        # Restore leading dims
        out = x.view(*lead_shape, *(self.output_shape or x.shape[1:]))
        return out


obs_spec = AttrDict({'shape': [channels, 64, 64], 'mean': 0.5, 'stddev': 0.5, 'low': 0, 'high': 1})
action_spec = AttrDict({'shape': obs_spec.shape, 'discrete_bins': None, 'low': -1, 'high': 1, 'discrete': False})

encoder = CNNEncoder(obs_spec, standardize=False, Eyes=nn.Identity)
actor = EnsemblePiActor(encoder.repr_shape, 100, -1, action_spec, trunk=Utils.Rand, Pi_head=Generator, ensemble_size=1,
                        lr=lr, optim={'_target_': 'Adam', 'betas': [beta1, 0.999]}).to(device)
critic = EnsembleQCritic(encoder.repr_shape, 100, -1, action_spec, Q_head=Discriminator, ensemble_size=1,
                         ignore_obs=True, lr=lr,
                         optim={'_target_': 'Adam', 'betas': [beta1, 0.999]}).to(device)


optimizerG = optim.Adam(actor.parameters(), lr=lr, betas=(beta1, 0.999))
criterion = nn.BCELoss()


for epoch in range(num_epochs):
    for i, (obs, *_) in enumerate(dataloader):

        # Discriminate real
        obs = obs.to(device)
        obs = encoder(obs)
        action = actor(obs).mean
        reward = torch.ones((len(obs), 1)).to(obs)
        critic_loss = QLearning.ensembleQLearning(critic, actor, obs, obs.view_as(action), reward, 1, torch.ones(0), 1)

        # Discriminate plausible
        reward = torch.zeros_like(reward)
        critic_loss += QLearning.ensembleQLearning(critic, actor, obs, action, reward, 1, torch.ones(0), 1)

        Utils.optimize(critic_loss, critic)

        # Generate
        action = actor(obs).mean.view_as(obs)
        Qs = critic(obs, action).view(-1, 1)
        Q_target = torch.ones_like(reward)
        actor_loss = criterion(Qs, Q_target)

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
action = actor(obs).mean.view_as(obs)
plt.imshow(np.transpose(vutils.make_grid(action[:64].detach(), padding=2, normalize=True).cpu(), (1, 2, 0)))

path = Path('./Benchmarking/DCGAN/AC2Agent/classify/CelebA_1_Video_Image')
path.mkdir(parents=True, exist_ok=True)
plt.savefig(path / 'generated.png')

plt.close()
