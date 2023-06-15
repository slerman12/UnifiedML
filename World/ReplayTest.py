# Template created by Sam Lerman, slerman@ur.rochester.edu.
import time

from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor, Normalize, Compose

import torch
from torch import nn
from torch.optim import SGD
import torch.multiprocessing as mp

from World.Memory import Mem, Batch
from World.Replay import Replay

# Start learning and evaluating
if __name__ == '__main__':
    mp.set_start_method('spawn')

    epochs = 10
    batch_size = 32
    lr = 1e-2
    device = 'cpu'

    # Pre-process
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # Get data
    train_dataset = mnist.MNIST(root='./', train=True, transform=data_transform, download=True)
    test_dataset = mnist.MNIST(root='./', train=False, transform=data_transform, download=True)

    # Divide data into batches
    # train_loader = Replay(batch_size=batch_size, dataset=train_dataset, reload=False, device=device)
    train_loader = Replay(batch_size=batch_size, dataset='MNIST', reload=False, ram_capacity=1e6, device=device)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # The neural network
    model = nn.Sequential(nn.Flatten(),  # Linear layers accept 1D inputs, so flatten the 2D RGB/or grayscale images
                          nn.Linear(784, 128), nn.ReLU(),  # MNIST images are grayscale with height x width = 28 x 28 = 784
                          nn.Linear(128, 64), nn.ReLU(),  # Linear layer (input size -> output size) followed by ReLU
                          nn.Linear(64, 10))  # MNIST has 10 predicted classes

    model.to(device)

    # The loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optim = SGD(model.parameters(), lr=lr)

    correct = total = 0

    model.train()

    means = []
    batches = mp.Manager().list()
    clock = time.time()

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if isinstance(train_loader, Replay):
                x, y = batch.obs.to(device), batch.label.to(device)

                # Add norm
                x = (x - 0.1307) / 0.3081
            else:
                # batches.append(Batch({str(j): Mem(m[None, :], f'./{epoch}_{i}_{j}').shared()
                #                       for j, m in enumerate(batch)}))
                # batch = batches.pop()
                #
                # x, y = torch.as_tensor(batch['0'].mem[0]).to(device), torch.as_tensor(batch['1'].mem[0]).to(device)

                x, y = batch[0].to(device), batch[1].to(device)

            # means.extend([(x_.mean().item(), y_.item()) for x_, y_ in zip(x, y)])

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total += y.shape[0]

            if i % 1000 == 0:
                # Note: The epoch train_loader.epoch is out of sync towards the end for Replay
                # print('Epoch: {}, Training Accuracy: {}/{} ({:.0f}%)'.format(epoch, correct, total,
                #                                                              100. * correct / total))

                correct = total = 0

            # Learn!
            optim.zero_grad()
            loss.backward()
            optim.step()

        correct = total = 0  # Reset score statistics

        model.eval()

        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            y_pred = model(x).detach()

            correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total += y.shape[0]

        print('Elapsed: {}, Epoch: {}, Evaluation Accuracy: {}/{} ({:.0f}%)'.format(time.time() - clock, epoch, correct,
                                                                                    total, 100. * correct / total))
