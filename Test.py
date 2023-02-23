# from torch.utils.data import DataLoader
# from torchvision.datasets.mnist import MNIST
# from torchvision.transforms import ToTensor
#
#
# dataset = MNIST('./', download=True, transform=ToTensor())
#
# dataset = DataLoader(dataset=dataset,
#                      pin_memory=True)  # pin_memory triggers CUDA error
#
# for _ in dataset:
#     continue

import torch
from torch import nn
from torch.optim import SGD

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

mse = nn.MSELoss()

a = torch.rand([10])
b = torch.rand([10])

model1 = nn.Linear(10, 10)
model2 = nn.Linear(10, 10)

optim = SGD(list(model1.parameters()) + list(model2.parameters()), lr=1e-4)

y1 = model1(a)
y2 = model2(b)

ones = torch.ones([20])
mse(torch.cat([y1, y2], 0), ones).backward()
grad1 = model1.weight.grad
grad2 = model2.weight.grad

optim.zero_grad()
y1 = model1(a)
y2 = model2(b)
ones = torch.ones([10])
((mse(y1, ones) + mse(y2, ones)) / 2).backward()

assert torch.allclose(model1.weight.grad, grad1)
assert torch.allclose(model2.weight.grad, grad2)






# Full reproducible playground to test this yourself
