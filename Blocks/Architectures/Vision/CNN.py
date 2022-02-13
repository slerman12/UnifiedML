import torch
from torch import nn

import Utils


class CNN(nn.Module):
    def __init__(self, obs_shape=torch.Size([3, 84, 84]), out_channels=32, depth=3, flatten=True, out_dim=None):
        super().__init__()

        self.obs_shape = obs_shape
        in_channels = obs_shape[0]

        self.CNN = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(in_channels if i == 0 else out_channels,
                                      out_channels, 3, stride=2 if i == 0 else 1),
                            nn.ReLU()) for i in range(depth + 1)],
            nn.Flatten() if flatten else nn.Identity(),
        )

        self.out_dim = out_dim

        if out_dim is not None:
            height, width = Utils.cnn_output_shape(*obs_shape[1:], self.CNN)

            self.projection = nn.Sequential(
                nn.Linear(out_channels * height * width, out_dim),
                nn.ReLU(inplace=True))

        self.apply(Utils.weight_init)

    def forward(self, *x):
        # Optionally append context to channels assuming dimensions allow
        if len(x) > 1:
            context = [c.reshape(x[0].shape[0], c.shape[-1], 1, 1).expand(-1, -1, *self.obs_shape[1:])
                       for c in x[1:]]
            x = torch.cat([x[0], *context], 1)

        out = self.CNN(x.view(self.obs_shape))

        if self.out_dim is not None:
            out = self.projection(out)

        return out

