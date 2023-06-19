# Template created by Sam Lerman, slerman@ur.rochester.edu.
import time

from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor, Normalize, Compose

import torch
from torch import nn
from torch.optim import SGD
import torch.multiprocessing as mp

from Hyperparams.minihydra import get_args, instantiate, interpolate
from Utils import init
from World.Memory import Mem, Batch
from World.Replay import Replay


import time


class Profiler:
    def __init__(self):
        self.starts = {}
        self.profiles = {}
        self.counts = {}

    def start(self, name):
        self.starts[name] = time.time()

    def stop(self, name):
        if name in self.profiles:
            self.profiles[name] += time.time() - self.starts[name]
            self.counts[name] += 1
        else:
            self.profiles[name] = time.time() - self.starts[name]
            self.counts[name] = 1

    def print(self):
        for name in self.profiles:
            print(name, ':', self.profiles[name] / self.counts[name])
        self.profiles.clear()
        self.counts.clear()

profiler = Profiler()


@get_args(source='Hyperparams/args.yaml')
def agent(args):
    # Set random seeds, device
    init(args)

    # Train, test environments
    generalize = instantiate(args.environment, train=False, seed=args.seed + 1234)

    for arg in ('obs_spec', 'action_spec', 'evaluate_episodes'):
        if hasattr(generalize.env, arg):
            setattr(args, arg, getattr(generalize.env, arg))
    interpolate(args)  # Update args
    return instantiate(args.agent)


# Start learning and evaluating
if __name__ == '__main__':
    mp.set_start_method('spawn')

    epochs = 10
    batch_size = 32
    lr = 1e-2
    device = 'cpu'

    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    train_dataset = mnist.MNIST(root='./', train=True, transform=data_transform, download=True)
    test_dataset = mnist.MNIST(root='./', train=False, transform=data_transform, download=True)

    # train_loader = Replay(batch_size=batch_size, dataset=train_dataset, reload=False, device=device)
    train_loader = Replay(batch_size=batch_size, dataset='MNIST', device=device)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.agent = agent()

        def forward(self, x):
            return self.agent.actor(self.agent.encoder(x)).All_Qs.mean(1)

    # model = nn.Sequential(nn.Flatten(),
    #                       nn.Linear(784, 128), nn.ReLU(),
    #                       nn.Linear(128, 64), nn.ReLU(),
    #                       nn.Linear(64, 10))
    model = Model()

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = SGD(model.parameters(), lr=lr)

    correct = total = 0

    model.train()

    means = []
    batches = mp.Manager().list()
    clock = time.time()

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            profiler.start('learn')
            if isinstance(train_loader, Replay):
                # x, y = batch.obs, batch.label.view(-1)
                x, y = batch.obs, batch.label

                # Add norm
                # x = (x / 255 - 0.1307) / 0.3081
                # x = (x - 0.1307) / 0.3081
            else:
                # Important tea
                # batches.append(Batch({str(j): Mem(m[None, :], f'./{epoch}_{i}_{j}').shared()
                #                       for j, m in enumerate(batch)}))
                # batch = batches.pop()
                #
                # x, y = torch.as_tensor(batch['0'].mem[0]).to(device), torch.as_tensor(batch['1'].mem[0]).to(device)

                x, y = batch[0].to(device), batch[1].to(device)

            # means.extend([(x_.mean().item(), y_.item()) for x_, y_ in zip(x, y)])

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            # correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            # total += y.shape[0]
            #
            # if i % 1000 == 0:
            #     # Note: The epoch train_loader.epoch is out of sync towards the end for Replay
            #     # print('Epoch: {}, Training Accuracy: {}/{} ({:.0f}%)'.format(epoch, correct, total,
            #     #                                                              100. * correct / total))
            #
            #     correct = total = 0

            # Learn!
            optim.zero_grad()
            loss.backward()
            optim.step()
            profiler.stop('learn')

        correct = total = 0  # Reset score statistics

        model.eval()

        for _, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            y_pred = model(x).detach().view(-1, 10)

            correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total += y.shape[0]
            profiler.print()
        print('Elapsed: {}, Epoch: {}, Evaluation Accuracy: {}/{} ({:.0f}%)'.format(time.time() - clock, epoch, correct,
                                                                                    total, 100. * correct / total))
