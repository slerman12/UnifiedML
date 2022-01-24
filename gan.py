# prerequisites
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import functional as FF
from torchvision.utils import save_image

from Blocks.Actors import GaussianActorEnsemble
from Blocks.Critics import EnsembleQCritic

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 256


# MNIST Dataset
class Transform:
    def __call__(self, sample):
        sample = FF.to_tensor(sample) * 2 - 1
        # sample *= 255  # Encoder expects pixels  # TODO maybe reconfigure that
        # mean = stddev = [0.5] * sample.shape[0]  # Depending on num channels
        # sample = F.normalize(sample, mean, stddev)  # Generic normalization
        return sample


transform = Transform()


train_dataset = datasets.MNIST(root='./Datasets/ReplayBuffer/Classify/MNIST_Train', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./Datasets/ReplayBuffer/Classify/MNIST_Eval', train=False, transform=transform, download=False)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)


class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim + 100, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, *x):
        x = torch.cat(x, -1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        # return torch.sigmoid(self.fc4(x))
        return self.fc4(x)


# build network
z_dim = 100
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

# G = Generator(g_input_dim = z_dim, g_output_dim = mnist_dim).to(device)
G = GaussianActorEnsemble([z_dim], 512, 256, mnist_dim, 1).to(device)
# D = Discriminator(mnist_dim).to(device)
D = EnsembleQCritic([z_dim], 1024, 512, mnist_dim).to(device)

# loss
criterion = nn.MSELoss()

# optimizer
lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr = lr)
D_optimizer = optim.Adam(D.parameters(), lr = lr)


def D_train(x):
    # =======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    z = torch.randn(bs, z_dim).to(device)

    D_output = torch.min(D(x_real, z).Qs, 0)[0]
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    x_fake, y_fake = G(z).mean[:, 0], torch.zeros(bs, 1).to(device)

    D_output = torch.min(D(x_fake, z).Qs, 0)[0]
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return  D_loss.data.item()


def G_train(x):
    # =======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(bs, z_dim).to(device)
    y = torch.ones(bs, 1).to(device)

    G_output = G(z).mean[:, 0]
    D_output = torch.min(D(G_output, z).Qs, 0)[0]
    G_loss = -D_output.mean()

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


n_epoch = 200
for epoch in range(1, n_epoch+1):
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        (epoch), n_epoch, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

with torch.no_grad():
    test_z = torch.randn(bs, z_dim).to(device)
    generated = G(test_z).mean[:, 0]

    Path('./Benchmarking/g/g/g/g/').mkdir(exist_ok=True, parents=True)
    save_image(generated.view(generated.size(0), 1, 28, 28), './Benchmarking/g/g/g/g/sample_' + '.png')