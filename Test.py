from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor


dataset = MNIST('./', download=True, transform=ToTensor())

dataset = DataLoader(dataset=dataset,
                     pin_memory=True)  # pin_memory triggers CUDA error

for _ in dataset:
    continue
