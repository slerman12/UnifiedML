import math

import torch
from torch import nn

import Utils


class CNN(nn.Module):
    def __init__(self, input_shape, out_channels=32, depth=3, batch_norm=False, last_relu=True,
                 kernel_size=3, stride=2, padding=0, dilation=1, groups=1, bias=True, output_dim=None):
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = [input_shape]

        in_channels, *_ = self.input_shape = torch.Size(input_shape)

        self.CNN = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(in_channels if i == 0 else out_channels,
                                      out_channels, kernel_size, stride=stride if i == 0 else 1,
                                      padding=padding, dilation=dilation, groups=groups, bias=bias),
                            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                            nn.ReLU() if i < depth or last_relu else nn.Identity()) for i in range(depth + 1)],
        )

        if output_dim is not None:
            shape = Utils.cnn_feature_shape(input_shape, self.CNN)

        self.project = nn.Identity() if output_dim is None \
            else nn.Sequential(nn.Flatten(), nn.Linear(math.prod(shape), 50), nn.ReLU(), nn.Linear(50, output_dim))

        self.apply(Utils.weight_init)

    def repr_shape(self, *_):
        return Utils.cnn_feature_shape(_, self.CNN, self.project)

    def forward(self, *x):
        lead_shape, x = broadcast(self.input_shape, x)

        x = self.CNN(x)
        x = self.project(x)

        # Restore lead dims
        out = x.view(*lead_shape, *x.shape[1:])

        return out


class Conv(CNN):
    def __init__(self, input_shape, out_channels=32, kernel_size=3, stride=2, padding=0, dilation=1, bias=True):
        super().__init__(input_shape, out_channels, depth=1, last_relu=False, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, bias=bias)


def broadcast(input_shape, x):
    """
    Accepts multiple inputs in a list and various shape possibilities, infers batch dims.
    Handles broadcasting as follows:

        1. Use raw input if input matches pre-specified input dims
        2. Otherwise, try un-flatten of last dim into expected spatial dims - depends on pre-specified input dims
        3. Or, create spatial dims via repetition of last dim = in channels - depends on pre-specified input dims
        4. Altogether ignore if empty
        5. Finally, concatenate along channels

    Allows images to be paired with lower-dim contexts or other images, inferring if lower-dim or even flattened.
    """

    _, *spatial_shape = input_shape

    # Lead shape for collapsing batch dims
    for input in x:
        if input.shape[-len(input_shape):] == input_shape:
            lead_shape = input.shape[:-len(input_shape)]
            break
        lead_shape = input.shape[:-1]

    # Broadcast as above
    x = torch.cat(
        [input if input.shape[-len(input_shape):] == input_shape
         else input.unflatten(-1, spatial_shape) if input.shape[-1] % math.prod(spatial_shape) == 0
        else input.view(*input.shape, *[1] * len(spatial_shape)).expand(*input.shape, *spatial_shape)
         for input in x if input.nelement() > 0], dim=-len(input_shape))

    # Collapse batch dims; operate on remaining dims
    x = x.view(-1, *x.shape[-len(input_shape):])

    return lead_shape, x


class AvgPool(nn.Module):
    def __init__(self, **_):
        super().__init__()

    def repr_shape(self, dim, *_):
        return dim,

    def forward(self, input):
        return input.mean(tuple(range(2, len(input.shape))))
