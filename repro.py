import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2))

    def forward(self, x):
        return self.CNN(x)


# cnn = CNN()
cnn = CNN().to('cuda')
x = torch.full([128, 1, 28, 28], float('nan')).to('cuda')
print(x.shape, torch.isnan(x).all())
y = cnn(x)
print(y.shape, torch.isnan(y).all())
