import math

import torch
from torch import nn

import Utils


class CNN(nn.Module):
    def __init__(self, input_shape, out_channels=32, depth=3, batch_norm=False, output_dim=None):
        super().__init__()

        self.input_shape = torch.Size(input_shape)
        in_channels = input_shape[0]

        self.CNN = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(in_channels if i == 0 else out_channels,
                                      out_channels, 3, stride=2 if i == 0 else 1),
                            nn.BatchNorm2d(self.out_channels) if batch_norm else nn.Identity(),
                            nn.ReLU()) for i in range(depth + 1)],
        )

        self.output_dim = output_dim

        if output_dim is not None:
            height, width = Utils.cnn_feature_shape(*input_shape[1:], self.CNN)

            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(out_channels * height * width, output_dim),
                nn.ReLU(inplace=True))

        self.apply(Utils.weight_init)

    def feature_shape(self, h, w):
        return Utils.cnn_feature_shape(h, w, self.CNN)

    def forward(self, *x):
        # Optionally append context to channels assuming dimensions allow
        if len(x) > 1:
            x = [context.view(*context.shape[:-1], -1, *self.input_shape[1:]) if context.shape[-1]
                                                                                 % math.prod(self.input_shape) == 0
                 else context.view(*context.shape[:-1], -1, 1, 1).expand(*context.shape[:-1], -1, *self.input_shape[1:])
                 for context in x if len(context.shape) < 4 and context.shape[-1]]
        x = torch.cat(x, -3)

        # Conserve leading dims
        lead_shape = x.shape[:-3]

        # Operate on last 3 dims
        x = x.view(-1, *self.input_shape)

        out = self.CNN(x)

        # Restore shape
        out = out.view(*lead_shape, out.shape[-3:])

        if self.output_dim is not None:
            out = self.projection(out)

        return out


class SimpleDecoder(nn.Module):
    def __init__(self, out_shape, depth=3):
        super().__init__()

        channels_in = 3
        channels_out = out_shape[0]

        self.CNN = nn.Sequential(*[nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(channels_in // 2 ** (i - 1), channels_out if i == depth else channels_in // 2 ** i, 3, padding=1),
            nn.GLU(1)) for i in range(depth + 1)])

    def forward(self, x):
        return self.CNN(x)
