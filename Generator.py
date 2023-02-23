# Template created by Sam Lerman, slerman@ur.rochester.edu.
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.Generator = nn.Sequential(
            # (64 * 8) x 4 x 4
            nn.ConvTranspose2d(3, 64 * 8, 4, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(inplace=True),

            # (64 * 4) x 8 x 8
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(inplace=True),

            # (64 * 2) x 16 x 16
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(inplace=True),

            # 64 x 32 x 32
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 3 x 64 x 64
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
        )

        self.apply(weight_init)

    def forward(self, x):
        return self.Generator(x)


# Initializes model weights a la normal
def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)) or isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
