import torch

from Blocks.Encoders import CNNEncoder


# cnn = CNN()
cnn = CNNEncoder((1, 28, 28), optim_lr=0.0001).to('cuda')
x = torch.full([128, 1, 28, 28], float('nan')).to('cuda')
print(x.shape, torch.isnan(x).all())
y = cnn(x)
print(y.shape, torch.isnan(y).all())
