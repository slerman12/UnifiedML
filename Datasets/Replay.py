import timeit

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tensordict import TensorDict


class SimpleDataset(Dataset):
    def __init__(self, data, device):
        # We split into a list since it is faster to dataload (fair comparison vs others)
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].to(self.device)


def run():
    d = torch.rand(1000, 50, 2)
    dataset = SimpleDataset(d, 'cuda')
    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, collate_fn=torch.stack)
    next(iter(dataloader))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    print(timeit.timeit(run, number=1))
