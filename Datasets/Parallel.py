import torch
from torch import nn


class Parallelize(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.replicas = tuple(model.to(torch._utils._get_device_index(device, True))
                              for device in torch._utils._get_all_device_indices())

    def forward(self, *args, **kwargs):
        return torch.nested.nested_tensor([model(*args, *kwargs) for model in self.models])

    def __getattr__(self, key):
        return getattr(self.replicas[0], key)

    def __setattr__(self, key, value):
        for replica in self.replicas:
            return setattr(replica, key, value)
