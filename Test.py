from torchvision.datasets.mnist import MNIST

dataset = MNIST('./')

for x, y in dataset:
    continue
