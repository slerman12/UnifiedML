import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch import multiprocessing as mp


class SimpleDataset(Dataset):
    def __init__(self, data, device, queue):
        # We split into a list since it is faster to dataload (fair comparison vs others)
        self.data = data
        self.device = device
        self.queue = queue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.device == 'cuda':
            data.to(self.device).share_memory_()
        self.queue.put(data)


def run():
    prefetch_queue = mp.Queue()
    d = torch.rand(1000, 50, 2)
    dataset = SimpleDataset(d, 'mps', prefetch_queue)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, collate_fn=none)
    next(iter(dataloader))
    while not prefetch_queue.empty():
        print(prefetch_queue.get().shape)


def none(*x, **y):
    return


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    run()
    # print(timeit.timeit(run, number=1))
