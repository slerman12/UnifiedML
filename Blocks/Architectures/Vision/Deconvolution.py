import math

from torch import nn

import Utils

from Blocks.Architectures.Vision.CNN import cnn_broadcast


class CNNTranspose(nn.Module):
    """
    A Mini U-Net. Partially adaptive: Adds extra channel/spatial dim(s) to input up to 2d.
    TODO Maybe move to CNN and reuse most of template
    """
    def __init__(self, input_shape, out_channels=32, output_shape=None):
        super().__init__()

        self.input_shape, self.output_shape = Utils.to_tuple(input_shape), Utils.to_tuple(output_shape)
        # TODO Maybe broadcast to at least 1D, but then would have to make all architectures inherently adaptive?
        #     Ord can leave as is, knowing that the corner case of using it as Eyes on 1d input wouldn't work
        self.input_shape = (1,) * (3 - len(self.input_shape)) + tuple(self.input_shape)  # Broadcast input to 2D
        in_channels = self.input_shape[0]

        self.CNNTranspose = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 0, bias=False),
            nn.Tanh()
        )

        self.repr = nn.Identity()  # Optional output projection

        if output_shape is not None:
            shape = self.repr_shape(*self.input_shape)

            # Optional output projection
            self.repr = nn.Sequential(nn.Flatten(), nn.Linear(math.prod(shape), math.prod(output_shape)))

        self.apply(Utils.weight_init)

    def repr_shape(self, *_):
        return Utils.repr_shape(self.input_shape, self.CNNTranspose, self.repr)

    def forward(self, *x):
        # Concatenate inputs along channels assuming dimensions allow, broadcast across many possibilities
        lead_shape, x = cnn_broadcast(self.input_shape, x)

        x = self.CNNTranspose(x)
        x = self.repr(x)  # Optional output projection

        # Restore leading dims
        out = x.view(*lead_shape, *(self.output_shape or x.shape[1:]))
        return out
