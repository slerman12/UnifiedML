from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm

dataset = MNIST('./', download=True)

dataset = DataLoader(dataset=dataset,
                     batch_size=256,
                     shuffle=True,
                     num_workers=8,
                     pin_memory=True)

for i, (x, y) in tqdm(enumerate(dataset)):
    continue
