# Template created by Sam Lerman, slerman@ur.rochester.edu.
import time

from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor, Normalize, Compose  # Can pre-process images

import torch  # Pytorch!
from torch import nn  # Neural networks
from torch.optim import SGD  # Stochastic gradient descent -- optimize the neural networks
import torch.multiprocessing as mp

from World.Memory import Mem, Batch
from World.Replay import Replay

# Start learning and evaluating
if __name__ == '__main__':
    mp.set_start_method('spawn')

    epochs = 10  # How many times to iterate through the full data for training
    batch_size = 32  # How many data-points to feed into the neural network at a time
    lr = 1e-2  # Learning rate -- controls the magnitude of gradients
    device = 'cpu'  # Can write 'cuda' for GPU if you have one

    # Pre-process
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # Get data
    train_dataset = mnist.MNIST(root='./', train=True, transform=data_transform, download=True)
    test_dataset = mnist.MNIST(root='./', train=False, transform=data_transform, download=True)

    # Divide data into batches
    # train_loader = Replay(batch_size=batch_size, dataset=train_dataset, reload=False, device=device)
    train_loader = Replay(batch_size=batch_size, dataset='MNIST', reload=False, ram_capacity=0, device=device)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # The neural network
    model = nn.Sequential(nn.Flatten(),  # Linear layers accept 1D inputs, so flatten the 2D RGB/or grayscale images
                          nn.Linear(784, 128), nn.ReLU(),  # MNIST images are grayscale with height x width = 28 x 28 = 784
                          nn.Linear(128, 64), nn.ReLU(),  # Linear layer (input size -> output size) followed by ReLU
                          nn.Linear(64, 10))  # MNIST has 10 predicted classes

    model.to(device)  # Move model to device (e.g. CPU, GPU)

    # The loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optim = SGD(model.parameters(), lr=lr)

    correct = total = 0

    # Just sets model.training to True. Some neural networks behave differently during training (e.g. nn.Dropout).
    # Here, makes no difference. Just convention
    model.train()

    means = []
    batches = mp.Manager().list()
    clock = time.time()

    # Train on the training data
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

            y_pred = model(x)  # Predict a class
            loss = loss_fn(y_pred, y)  # Compute error

            # Tally scores
            correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total += y.shape[0]

            # Print scores
            if i % 1000 == 0:
                # Note: The epoch train_loader.epoch is out of sync towards the end for Replay
                # print('Epoch: {}, Training Accuracy: {}/{} ({:.0f}%)'.format(epoch, correct, total,
                #                                                              100. * correct / total))

                correct = total = 0

            # Optimize the neural network - learn!
            optim.zero_grad()  # Resets model's internal gradients to zero
            loss.backward()  # Adds the new gradients into memory (by computing them via the backpropagation function)
            optim.step()  # Steps those gradients on the model. Independent since you might want to backprop multiple losses

        correct = total = 0  # Reset score statistics

        model.eval()  # Sets model.training to False

        # Evaluate scores on the evaluation data
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            y_pred = model(x).detach()

            correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total += y.shape[0]

        print('Elapsed: {}, Epoch: {}, Evaluation Accuracy: {}/{} ({:.0f}%)'.format(time.time() - clock, epoch, correct,
                                                                                    total, 100. * correct / total))
