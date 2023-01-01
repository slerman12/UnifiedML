from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import functional as F
from tqdm import tqdm


class Transform:
    def __call__(self, x):
        return F.to_tensor(x)


dataset = MNIST('./', download=True, transform=Transform())

dataset = DataLoader(dataset=dataset)

for i, (x, y) in tqdm(enumerate(dataset)):
    continue
