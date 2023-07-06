import multiprocessing as mp
import os
from abc import abstractmethod

import torch


NCORE = 4


def process(q, iolock):
    from time import sleep
    while True:
        stuff = q.get()
        if stuff is None:
            break
        with iolock:
            print("processing", stuff)
        sleep(stuff)


class PersistentWorker:
    def __init__(self, worker, num_workers, map_args=True, **kwargs):
        self.main_worker = os.getpid()

        self.worker = worker
        self.num_workers = num_workers

        # For sending data to and from workers
        self.main_pipes, self.worker_pipes = zip(*[mp.Pipe() for _ in range(num_workers)])

        self.dones = [torch.tensor(False).share_memory_() for _ in range(num_workers)]
        self.dones[-1][...] = True

        self.map_args = map_args  # Can assume args or at least signal to go

        self.__dict__.update(kwargs)

    def __call__(self):
        while True:
            if (self.worker_pipes[self.worker].poll() or not self.map_args) \
                    and not self.main_pipes[self.worker].poll() \
                    and self.dones[self.worker - 1 if self.worker else -1]:
                args = self.worker_pipes[self.worker].recv() if self.map_args else ()
                outs = self.target(*args)
                self.worker_pipes[self.worker].send(outs)
                self.dones[self.worker - 1 if self.worker else -1][...] = False
                self.dones[self.worker][...] = True  # FULLY SEQUENTIAL!!! Iterate through all pipes and order by pipe

    @abstractmethod
    def target(self, *args):
        pass

    def map(self, args):
        assert self.main_worker == os.getpid(), 'Only main worker can call consume.'

        if self.map_args:
            for i in range(self.num_workers):
                self.main_pipes[i].send(args[i])

        outs = []
        while len(outs) < self.num_workers:
            if self.main_pipes[len(outs)].poll():
                outs.append(self.main_pipes[len(outs)].recv())

        return outs