from pathlib import Path

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision.transforms import functional as FF
from torchvision.utils import save_image

from Blocks.Actors import GaussianActorEnsemble
from Blocks.Critics import EnsembleQCritic
from Blocks.Augmentations import RandomShiftsAug

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

z_dim = 1568
lr = 0.0002

G = GaussianActorEnsemble([z_dim], 50, 1024, mnist_dim, 1, optim_lr=lr).to(device)
D = EnsembleQCritic([z_dim], 50, 1024, mnist_dim, optim_lr=lr).to(device)
aug = RandomShiftsAug(4)

loss = nn.MSELoss()


def D_train(x, z):
    D.zero_grad()

    x, y = x.flatten(-3).to(device), torch.ones(x.shape[0], 1).to(device)

    half = x.shape[0] // 2
    x[:half], y[:half] = G(z[:half]).mean[:, 0], 0

    D_output = D(z, x)

    D_loss = loss(D_output.Qs, y.expand_as(D_output.Qs))

    D_loss.backward()
    D.optim.step()

    return D_loss


def G_train(z):
    G.zero_grad()

    G_output = G(z).mean[:, 0]
    D_output = torch.min(D(z, G_output).Qs, 0)[0]
    G_loss = -D_output.mean()

    G_loss.backward()
    G.optim.step()

    return G_loss


epochs = 200
for epoch in range(1, epochs + 1):
    D_losses, G_losses = [], []
    for x, _ in train_loader:
        x = aug(x)
        z = torch.randn(x.shape[0], z_dim).to(device)
        D_losses.append(D_train(x, z).data.item())
        G_losses.append(G_train(z).data.item())

    print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
        epoch, epochs, torch.tensor(D_losses).mean(), torch.tensor(G_losses).mean()))

with torch.no_grad():
    z = torch.randn(batch_size, z_dim).to(device)
    generated = G(z).mean[:, 0]

    path = './Benchmarking/g/g/g/g/'
    Path(path).mkdir(exist_ok=True, parents=True)
    save_image(generated.view(generated.size(0), 1, 28, 28), path + 'sample.png')