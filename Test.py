from torchvision.datasets.mnist import MNIST
from tqdm import tqdm

dataset = MNIST('./', download=True)

for i, (x, y) in tqdm(enumerate(dataset)):
    continue
