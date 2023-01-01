from torchvision.datasets.mnist import MNIST

dataset = MNIST('./', download=True)

for i, (x, y) in enumerate(dataset):
    continue
