from pathlib import Path

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision.transforms import functional as FF
from torchvision.utils import save_image

from Blocks.Actors import GaussianActorEnsemble
from Blocks.Critics import EnsembleQCritic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 256


class Transform:
    def __call__(self, sample):
        return FF.to_tensor(sample) * 2 - 1


transform = Transform()

train_dataset = datasets.MNIST(root='./Datasets/ReplayBuffer/Classify/MNIST_Train', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./Datasets/ReplayBuffer/Classify/MNIST_Eval', train=False, transform=transform, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)

z_dim = 100
lr = 0.0002

G = GaussianActorEnsemble([z_dim], 512, 256, mnist_dim, 1, optim_lr=lr).to(device)
D = EnsembleQCritic([z_dim], 1024, 512, mnist_dim, optim_lr=lr).to(device)

loss = nn.MSELoss()


def D_train(x):
    D.zero_grad()

    x_real, y_real = x.view(-1, mnist_dim).to(device), torch.ones(x.shape[0], 1).to(device)

    z = torch.randn(x.shape[0], z_dim).to(device)
    shape = x.shape[0] // 2

    x_real[:shape], y_real[:shape] = G(z[:shape]).mean[:, 0], 0

    D_output = torch.min(D(z, x_real).Qs, 0)[0]

    D_loss = loss(D_output, y_real)

    D_loss.backward()
    D.optim.step()

    return D_loss.data.item()


def G_train(x):
    G.zero_grad()

    z = torch.randn(batch_size, z_dim).to(device)

    G_output = G(z).mean[:, 0]
    D_output = torch.min(D(z, G_output).Qs, 0)[0]
    G_loss = -D_output.mean()

    G_loss.backward()
    G.optim.step()

    return G_loss.data.item()


epochs = 200
for epoch in range(1, epochs + 1):
    D_losses, G_losses = [], []
    for batch_idx, (x, _) in enumerate(train_loader):
        D_losses.append(D_train(x))
        G_losses.append(G_train(x))

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        epoch, epochs, torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses))))

with torch.no_grad():
    generated = G(torch.randn(batch_size, z_dim).to(device)).mean[:, 0]

    Path('./Benchmarking/g/g/g/g/').mkdir(exist_ok=True, parents=True)
    save_image(generated.view(generated.size(0), 1, 28, 28), './Benchmarking/g/g/g/g/sample_' + '.png')