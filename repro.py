import torch
from torch import nn


cnn = nn.Conv2d(1, 32, 3, stride=2).to('cuda')
x = torch.full([128, 1, 28, 28], float('nan')).to('cuda')
print(torch.isnan(x).all())
y = cnn(x)
print(torch.isnan(y).all())
