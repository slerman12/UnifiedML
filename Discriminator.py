# Template created by Sam Lerman, slerman@ur.rochester.edu.

from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.Discriminator = nn.Sequential(
            # 64 x 32 x 32
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (64 * 2) x 16 x 16
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (64 * 4) x 8 x 8
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (64 * 8) x 4 x 4
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 1 x 1 x 1
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        self.apply(weight_init)

    def forward(self, x):
        return self.Discriminator(x)


# Initializes model weights a la normal
def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)) or isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
