import torch
from torch import nn

import Utils


class CNN(nn.Module):
    def __init__(self, input_shape=torch.Size([3, 84, 84]), out_channels=32, depth=3, batch_norm=False,
                 flatten=True, output_dim=None):
        super().__init__()

        self.input_shape = input_shape
        in_channels = input_shape[0]

        self.CNN = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(in_channels if i == 0 else out_channels,
                                      out_channels, 3, stride=2 if i == 0 else 1),
                            nn.BatchNorm2d(self.out_channels) if batch_norm else nn.Identity(),
                            nn.ReLU()) for i in range(depth + 1)],
            nn.Flatten() if flatten else nn.Identity(),
        )

        self.output_dim = output_dim

        if output_dim is not None:
            height, width = Utils.cnn_output_shape(*input_shape[1:], self.CNN)

            self.projection = nn.Sequential(
                nn.Linear(out_channels * height * width, output_dim),
                nn.ReLU(inplace=True))

        self.apply(Utils.weight_init)

    def output_shape(self, h, w):
        return Utils.cnn_output_shape(h, w, self.CNN) if self.output_dim is None \
            else (1, self.output_dim)

    def forward(self, *x):
        # Optionally append context to channels assuming dimensions allow
        if len(x) > 1:
            x[1:] = [context.reshape(x[0].shape[0], context.shape[-1], 1, 1).expand(-1, -1, *self.input_shape[1:])
                     for context in x[1:]]

        x = torch.cat(x, 1)

        out = self.CNN(x.view(self.input_shape))

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
