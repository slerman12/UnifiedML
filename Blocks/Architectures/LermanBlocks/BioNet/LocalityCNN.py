import torch
from torch import nn, einsum
from torch.nn import functional as F

from Blocks.Architectures.Residual import Residual

import Utils


class Conv2DLocalized(nn.Module):
    def __init__(self, input_shape, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                 groups=1, bias=True, padding_mode='zeros'):
        super().__init__()

        def layer_norm():
            return nn.Sequential(Utils.ChannelSwap(),
                                 nn.LayerNorm(out_channels),
                                 Utils.ChannelSwap())

        in_channels, height, width = input_shape

        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                            groups, bias, padding_mode),
                                  nn.GELU(),
                                  layer_norm())

        height, width = Utils.cnn_layer_feature_shape(height, width, kernel_size,
                                                     stride, padding, dilation)

        self.shape = (out_channels, height, width)

        self.linear_W = nn.Parameter(torch.empty(height, width, out_channels, out_channels))

        self.linear_B = nn.Parameter(torch.empty(height, width, out_channels))

        self.ln = layer_norm()

    def feature_shape(self, h, w):
        return Utils.cnn_feature_shape(h, w, self.conv)

    def forward(self, input):
        x = self.conv(input)
        x = einsum('b c h w, h w d c, h w d -> b c h w', x, self.linear_W.to(input.device), self.linear_B.to(input.device))
        x = F.gelu(x)
        x = self.ln(x)
        return x


class LocalityCNN(nn.Module):
    def __init__(self, input_shape, out_channels=32, depth=3):
        super().__init__()

        self.trunk = Conv2DLocalized(input_shape, out_channels, (8, 8))

        self.CNN = nn.Sequential(
            *[Residual(Conv2DLocalized(self.trunk.shape, out_channels, (4, 4), padding='same'))
              for _ in range(depth)])

    def feature_shape(self, h, w):
        h, w = Utils.cnn_feature_shape(h, w, self.trunk)
        return Utils.cnn_feature_shape(h, w, self.CNN)

    def forward(self, x):
        return self.CNN(self.trunk(x))
