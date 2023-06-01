import torch
from torch import nn


# class Parallelize(nn.Module):
#     def __init__(self, module, coalesce=False):
#         super().__init__()
#
#         self.devices = torch._utils._get_all_device_indices()
#
#         self.replicas = nn.ModuleList([module.to(torch._utils._get_device_index(device, True))
#                                        for device in self.devices] if self.devices else [module])
#
#         self.streams = None
#         self.coalesce = coalesce
#
#         print(f'Parallelizing across {len(self.replicas) if self.devices else 0} cuda devices.')
#
#     def forward(self, *args):
#         if self.streams is None:
#             # Just-in-time since can't be pickled for EMA
#             self.streams = [torch.cuda.current_stream(device) for device in self.devices]
#
#         if len(self.replicas) > 1:
#             splits = [getattr(arg, 'splits') for arg in args] if hasattr(args[0], 'splits') \
#                 else []
#
#             if not splits:
#                 for i, arg in enumerate(args):
#                     quotient = len(arg) // len(self.devices)
#                     remainder = len(arg) % len(self.devices)
#
#                     split = [quotient] * (len(self.devices) + bool(remainder))
#                     split[-1] += remainder
#
#                     splits.append(torch.split(arg, split))
#
#             outs = []
#
#             for i, module in enumerate(self.replicas):
#                 with torch.cuda.stream(self.streams[i]):
#                     outs.append(module(*[split[i] for split in splits]).to(self.devices[0]))
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


class Parallelize(nn.Module):  # Slightly faster than DataParallel
    def __init__(self, module):
        super().__init__()

        self.devices = torch._utils._get_all_device_indices()

        self.replicas = nn.ModuleList([module.to(torch._utils._get_device_index(device, True))
                                       for device in self.devices] if self.devices else [module])

        self.streams = None

        print(f'Parallelizing across {len(self.replicas) if self.devices else 0} cuda devices.')

    def forward(self, *args):
        if self.streams is None:
            # Just-in-time since can't be pickled for EMA
            self.streams = [torch.cuda.current_stream(device) for device in self.devices]

        if len(self.replicas) > 1:
            splits = []

            for i, arg in enumerate(args):
                quotient = len(arg) // len(self.devices)
                remainder = len(arg) % len(self.devices)

                split = [quotient] * (len(self.devices) + bool(remainder))
                split[-1] += remainder

                splits.append(torch.split(arg, split))

            outs = []

            for i, module in enumerate(self.replicas):
                with torch.cuda.stream(self.streams[i]):
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
