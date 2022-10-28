import math

import torch
from torch import nn

import Utils


class CNN(nn.Module):
    def __init__(self, input_shape, out_channels=32, depth=3, batch_norm=False, last_relu=True,
                 kernel_size=3, stride=2, padding=0, output_dim=None):
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = [input_shape]

        in_channels, *_ = self.input_shape = torch.Size(input_shape)

        self.CNN = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(in_channels if i == 0 else out_channels,
                                      out_channels, kernel_size, stride=stride if i == 0 else 1,
                                      padding=padding),
                            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                            nn.ReLU() if i < depth or last_relu else nn.Identity()) for i in range(depth + 1)],
        )

        if output_dim is not None:
            shape = Utils.cnn_feature_shape(input_shape, self.CNN)

        self.project = nn.Identity() if output_dim is None \
            else nn.Sequential(nn.Flatten(), nn.Linear(math.prod(shape), 50), nn.ReLU(), nn.Linear(50, output_dim))

        self.apply(Utils.weight_init)

    def repr_shape(self, c, h, w):
        return Utils.cnn_feature_shape([c, h, w], self.CNN, self.project)

    def forward(self, *x):
        lead_shape, x = broadcast(self.input_shape, x)

        x = self.CNN(x)
        x = self.project(x)

        # Restore lead dims
        out = x.view(*lead_shape, *x.shape[1:])

        return out


# TODO Add to other Vision architectures
def broadcast(input_shape, x):
    """
    Accepts multiple inputs in a list and various shape possibilities, infers batch dims.
    Handles broadcasting as follows:

        1. Use raw input if matches pre-specified input dims
        2. Otherwise, try to un-flatten last dim of input into expected spatial dims - depends on given input dims
        3. Or, create spatial dims via repetition of last dim as channel dim - depends on given input dims
        4. Altogether ignore if empty
        5. Finally, concatenate along channels

    Allows images to be paired with lower-dim contexts or other images, inferring if lower-dim or even flattened.
    """

    _, *spatial_shape = input_shape
    concat = []
    had_spatial_dims = False

    for i, input in enumerate(x):
        tail_shape = input.shape[-len(input_shape):]

        raw = tail_shape == input_shape

        if input.nelement() > 0:
            if raw:
                concat.append(input)
                had_spatial_dims = True
                lead_shape = input.shape[:-len(input_shape)]
            else:
                unflatten = input.shape[-1] % math.prod(spatial_shape) == 0

                if unflatten:
                    concat.append(input.unflatten(-1, spatial_shape))
                else:
                    concat.append(input.view(*input.shape, *[1] * len(spatial_shape))
                                  .expand(*input.shape, *spatial_shape))

                if not had_spatial_dims:
                    lead_shape = input.shape[:-1]

    x = torch.concat(concat, dim=-len(input_shape))

    # x = torch.cat(
    #     [input if input.shape[-len(self.input_shape):] == self.input_shape
    #      else input.unflatten(-1, self.spatial_shape) if input.shape[-1] % math.prod(self.spatial_shape) == 0
    #     else input.view(*input.shape, *[1] * len(self.spatial_shape)).expand(*input.shape, *self.spatial_shape)
    #      for input in x if input.nelement() > 0], dim=-len(self.input_shape))

    # Operate on remaining dims
    x = x.view(-1, *x.shape[-len(input_shape):])

    return lead_shape, x


class AvgPool(nn.Module):
    def __init__(self, **_):
        super().__init__()

    def repr_shape(self, dim, *_):
        return dim,

    def forward(self, input):
        for _ in input.shape[2:]:
            input = input.mean(-1)
        return input
