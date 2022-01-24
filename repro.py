import torch
from torch import nn

from Blocks.Encoders import CNNEncoder


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2), nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1), nn.ReLU())

    def forward(self, x, *context):
        return self.CNN(x)


# cnn = CNN()
cnn = CNNEncoder((1, 28, 28)).to('cuda')
x = torch.full([128, 1, 28, 28], float('nan')).to('cuda')
print(x.shape, torch.isnan(x).all())
y = cnn(x)
print(y.shape, torch.isnan(y).all())
