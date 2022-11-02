import math

from torch import nn

import Utils
from Blocks.Architectures.Vision.CNN import broadcast


class CNNTranspose(nn.Module):
    def __init__(self, input_shape, out_channels=32, output_dim=None):
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
        )  # TODO Actor Generator needs to not project, just flatten - translate output_dim into deconv layers

        if output_dim is not None:
            shape = Utils.cnn_feature_shape(input_shape, self.CNNTranspose)  # TODO ConvTranspose2d support

        self.project = nn.Identity() if output_dim is None \
            else nn.Sequential(nn.Flatten(), nn.Linear(math.prod(shape), output_dim))

        self.apply(Utils.weight_init)

    def repr_shape(self, *_):
        return Utils.cnn_feature_shape(_, self.CNNTranspose, self.project)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = broadcast(self.input_shape, x)

        x = self.CNNTranspose(x)
        x = self.project(x)

        # Restore leading dims
        out = x.view(*lead_shape, *x.shape[1:])
        return out
