import torch
from torch import nn

# import os
# from timeit import timeit
#
#
# def func1():
#     arrays = [torch.ones([100000, 10]).to(device=f'cpu:{cpu}') for cpu in range(1, os.cpu_count())]
#     return list(map(torch.sum, arrays))
#
#
# def func2():
#     arrays = [torch.ones([100000, 10]).to(device=f'cpu:1') for _ in range(1, os.cpu_count())]
#     return list(map(torch.sum, arrays))
#
#
# print('Num workers', os.cpu_count() - 1)
# trials = 100
# timeit(func2, number=trials)
# print(timeit(func1, number=trials), 'parallel')
# print(timeit(func2, number=trials), 'not parallel')
# print(timeit(func1, number=trials), 'parallel')
# print(timeit(func2, number=trials), 'not parallel')
# print(timeit(func1, number=trials), 'parallel')
# print(timeit(func2, number=trials), 'not parallel')
# print(timeit(func1, number=trials), 'parallel')
# print(timeit(func2, number=trials), 'not parallel')
# Result: doesn't seem to work


# https://pytorch.org/docs/stable/notes/extending.html Extending torch with a Tensor-like type
# https://jaketae.github.io/study/pytorch-tensor/
# class Parallelize(nn.Module):
#     def __init__(self, module, coalesce=False):
#         super().__init__()
#
#         self.devices = torch._utils._get_all_device_indices()
#
#         self.replicas = nn.ModuleList([module.to(torch._utils._get_device_index(device, True))
#                                        for device in self.devices] if self.devices else [module])
#
#         self.coalesce = coalesce
#
#         print(f'Parallelizing across {len(self.replicas) if self.devices else 0} cuda devices.')
#
#     def forward(self, *args):
#         if len(self.replicas) > 1:
#             splits = [getattr(arg, 'splits') for arg in args] if hasattr(args[0], 'splits') \
#                 else []
#
#             if not splits:
#                 for i, arg in enumerate(args):
#                     quotient, remainder = divmod(len(arg), len(self.devices))
#
#                     split = [quotient] * (len(self.devices) + bool(remainder) - 1) + [quotient + remainder]
#
#                     splits.append(torch.split(arg, split))
#
#             outs = []
#
#             for i, module in enumerate(self.replicas):
#                 outs.append(module(*[split[i] for split in splits]).to(self.devices[0]))
#
#             if self.coalesce:
#                 return torch.concat(outs)
#             else:
#                 setattr(outs[0], 'splits', outs)
#                 return outs[0]  # Note: risks copy operations and so on
#         else:
#             return self.replicas[0](*args)


# Need
# 1. Modified Tensor data structure
# 2. Scatter-reduce on gradients OR device-level optimization via sharing gradients back to the respective device
# 3. Syncing on this shared pool across hog-wild nodes - RPC uploader


class Parallelize(nn.Module):  # Slightly faster than DataParallel  TODO Are the replicas independent?? Separate optim
    def __init__(self, module):
        super().__init__()

        self.devices = torch._utils._get_all_device_indices()

        # TODO is .to even in-place?
        self.replicas = nn.ModuleList([module.to(torch._utils._get_device_index(device, True))
                                       for device in self.devices] if self.devices else [module])  # In-place, useless

        print(f'Parallelizing across {len(self.replicas) if self.devices else 0} cuda devices.')

    def forward(self, *args):
        if len(self.replicas) > 1:
            splits = []

            for i, arg in enumerate(args):
                quotient, remainder = divmod(len(arg), len(self.replicas))

                split = [quotient] * (len(self.replicas) - 1) + [quotient + remainder]

                splits.append(torch.split(arg, split))

            outs = []

            for i, module in enumerate(self.replicas):
                outs.append(module(*[split[i] for split in splits]).to(self.devices[0]))

            outs = torch.concat(outs)
        else:
            outs = self.replicas[0](*args)

        return outs


# https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/parallel_apply.py
# class Parallelize(nn.Module):  # Note: Slower than DataParallel; would probably need cuda.stream
#     def __init__(self, module):
#         super().__init__()
#
#         self.devices = torch._utils._get_all_device_indices()
#
#         self.replicas = nn.ModuleList([module.to(torch._utils._get_device_index(device, True))
#                                        for device in self.devices] if self.devices else [module])
#
#         print(f'Parallelizing across {len(self.replicas) if self.devices else 0} cuda devices.')
#
#     def forward(self, *args):
#         if len(self.replicas) > 1:
#             splits = []
#
#             for i, arg in enumerate(args):
#                 quotient = len(arg) // len(self.devices)
#                 remainder = len(arg) % len(self.devices)
#
#                 split = [quotient] * (len(self.devices) + bool(remainder))
#                 split[-1] += remainder
#
#                 splits.append(split)
#
#             splits = [torch.split(arg, split) for arg, split in zip(args, splits)]
#             args = [[split[device] for split in splits] for device in range(len(self.devices))]
#
#         return torch.concat([module(*args[i]).to(self.devices[0])
#                              for i, module in enumerate(self.replicas)]) if len(self.replicas) > 1 \
#             else self.replicas[0](*args)
