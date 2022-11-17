import math

from torch import nn

import Utils
from Blocks.Architectures.Vision.CNN import broadcast


class CNNTranspose(nn.Module):
    def __init__(self, input_shape, out_channels=32, output_shape=None):
        super().__init__()

        if isinstance(input_shape, int):
            input_shape = [input_shape]

        if len(input_shape) == 1:
            input_shape += (1, 1)  # Assumes 2d

        in_channels, *_ = self.input_shape = input_shape

        self.CNNTranspose = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        # Output shape of a conv = (W - F + 2P)/S + 1

        self.output_shape = output_shape

        output_shape = (output_shape,) if isinstance(output_shape, int) else None if output_shape is None \
            else (1,) * (3 - len(output_shape)) + tuple(output_shape)  # Adapts to channels, height, and width

        # if output_shape is not None:
        #     shape = Utils.cnn_feature_shape(input_shape, self.CNNTranspose)  # TODO ConvTranspose2d support
        #
        # self.project = nn.Identity() if output_shape is None \
        #     else nn.Sequential(nn.Flatten(), nn.Linear(math.prod(shape), output_shape))  # TODO project

        # test
        self.project = nn.Identity() if output_shape is None \
            else nn.Sequential(nn.Flatten(), nn.Linear(math.prod([32, 16, 16]), Utils.prod(output_shape)))

        self.apply(Utils.weight_init)

    def repr_shape(self, *_):
        return self.output_shape or Utils.cnn_feature_shape(_, self.CNNTranspose)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = broadcast(self.input_shape, x)

        x = self.CNNTranspose(x)
        x = self.project(x)

        # Restore leading dims
        out = x.view(*lead_shape, *(self.output_shape or x.shape[1:]))
        return out
