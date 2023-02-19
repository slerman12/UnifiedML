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


obs_spec = AttrDict({'shape': [channels, 64, 64], 'mean': 0.5, 'stddev': 0.5, 'low': 0, 'high': 1})  # Can set mean, stddev
action_spec = AttrDict({'shape': obs_spec.shape, 'discrete_bins': None, 'low': -1, 'high': 1, 'discrete': False})

encoder = CNNEncoder(obs_spec, standardize=False, Eyes=nn.Identity)

actor = EnsemblePiActor(encoder.repr_shape, 100, -1, action_spec, trunk=Utils.Rand, Pi_head=Generator, ensemble_size=1,
                        lr=lr, optim={'_target_': 'Adam', 'betas': [beta1, 0.999]})
critic = EnsembleQCritic(encoder.repr_shape, 100, -1, action_spec, Q_head=Discriminator, ensemble_size=1,
                         ignore_obs=True, lr=lr,
                         optim={'_target_': 'Adam', 'betas': [beta1, 0.999]})  # Note: trunk_dim for example isn't necessary for generate=true


# TODO perhaps try torch.amax(x, dim=(2,3)) # Global maximum pooling
# TODO Perhaps init agent with .to(memory_format=torch.channels_last) and .half()

# TODO Augs, Compose


netG = actor.to(device)
netD = critic.to(device)

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Initialize BCELoss function
# criterion = nn.MSELoss()
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        logs = {}

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)

        obs = encoder(real_cpu)

        noise = torch.randn(b_size, z_dim, 1, 1, device=device)
        action = netG(noise).mean
        reward = torch.full((b_size, 1), real_label, dtype=torch.float, device=device)

        critic_loss = QLearning.ensembleQLearning(critic, actor, obs, obs.view_as(action), reward, 1, torch.ones(0),
                                                  1, logs=logs)

        reward = torch.full((b_size, 1), fake_label, dtype=torch.float, device=device)

        # Critic loss
        critic_loss += QLearning.ensembleQLearning(critic, actor, obs, action, reward, 1, torch.ones(0), 1, logs=logs)

        Utils.optimize(critic_loss, critic)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        netG.zero_grad()

        noise = torch.randn(b_size * 2, z_dim, 1, 1, device=device)
        fake = netG(noise).mean
        fake = fake.view(-1, *real_cpu.shape[1:])

        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(obs, fake).view(-1)
        # output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)  # TODO MSE better than nothing because diminishes gradients closer to 0, 1
        # errG = -output.log().mean()  # TODO Try
        # errG = -output.mean()  # TODO Try
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader), True, errG.item(), True, True, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(True)

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).mean.detach().cpu()
                # fake = netG(fixed_noise).detach().cpu()
                fake = fake.view(fake.shape[0], *real_cpu.shape[1:])
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()
path = Path('./Benchmarking/DCGAN/AC2Agent/classify/CelebA_1_Video_Image')
path.mkdir(parents=True, exist_ok=True)
plt.savefig(path / 'generated.png')

plt.close()
