from torchvision.datasets.mnist import MNIST

dataset = MNIST('./', download=True)

for x, y in dataset:
    continue
