import torch
from torch import nn


import torch
from torch import nn


class Parallelize(nn.Module):  # Note: Slower than DataParallel; would probably need cuda.stream
    def __init__(self, module):
        super().__init__()

        self.devices = torch._utils._get_all_device_indices()

        self.replicas = nn.ModuleList([module.to(torch._utils._get_device_index(device, True))
                                       for device in self.devices] if self.devices else [module])

        print(f'Parallelizing across {len(self.replicas) if self.devices else 0} cuda devices.')

    def forward(self, *args):
        if len(self.replicas) > 1:
            splits = []

            for i, arg in enumerate(args):
                quotient = len(arg) // len(self.devices)
                remainder = len(arg) % len(self.devices)

                split = [quotient] * (len(self.devices) + bool(remainder))
                split[-1] += remainder

                splits.append(split)

            splits = [torch.split(arg, split) for arg, split in zip(args, splits)]
            args = [[split[device] for split in splits] for device in range(len(self.devices))]

            streams = [torch.cuda.Stream() for _ in args]
            outs = []

            torch.cuda.synchronize()

            for i, module in enumerate(self.replicas):
                with torch.cuda.stream(streams[i]):
                    outs.append(module(*args[i]).to(self.devices[0]))

            torch.cuda.synchronize()

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
