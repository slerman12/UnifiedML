import time

import torch
from torch.nn import functional as F, Parameter


# Since write has btach-size limits without cloning, maybe don't need write. Maybe can use Parameters (even on CPU)
# and just read
# Maybe can no_grad the params except for the read ones? If the reference checks out
# Given N1 x ... x Nn x V1 x ... x Vm grid,
# B x M1 x ... x Mn x n window,
# B x M1 x ... x Mn source,
# writes source to grid at window
from torch.optim import Adam


def write(grid, window, src):
    value_shape = grid.shape[len(window.shape) - 2:]

    index = window_to_index(window, grid)

    index = index.flatten(0, 1)
    grid.view(-1, *value_shape).scatter_(0, index, src.view(-1, *value_shape))


# Given N1 x ... x Nn x V1 x ... x Vm grid,
# B x M1 x ... x Mn x n window,
# returns B x M1 x ... x Mn x V1 x ... x Vm memory
def lookup(grid, window):
    batch_dim = window.shape[0]
    value_shape = grid.shape[len(window.shape) - 2:]

    index = window_to_index(window, grid)

    out = grid.expand(batch_dim, *grid.shape).view(batch_dim, -1, *value_shape).gather(1, index)
    # out = grid.view(-1, *value_shape)[index]
    # print(out.shape)

    return out.view(*window.shape[:-1], *value_shape)


# Given B x M1 x ... x Mn x n window,
# return B x M1 * ... * Mn index
def window_to_index(window, grid, unique=False):
    axes = window.shape[-1]
    grid_size = grid.shape[0]
    value_shape = grid.shape[len(window.shape) - 2:]

    index_lead = torch.matmul(window, grid_size ** reversed(torch.arange(axes))).flatten(1)

    if unique:
        print(index_lead.shape)
        index_lead, inds, counts = torch.unique(index_lead, return_inverse=True, return_counts=True)
        print(index_lead, index_lead.shape, counts)

    index = index_lead.view(*index_lead.shape, *([1] * len(value_shape))).expand(*index_lead.shape, *value_shape)

    return index


# Given B x n coord,
# returns B x M1 x ... x Mn x n window
def coord_to_window(coord, window_size, grid_size):
    axes = coord.shape[1]
    range = torch.arange(window_size) - window_size // 2
    mesh = torch.stack(torch.meshgrid(*([range] * axes)), -1)
    window = coord.view(-1, *([1] * axes), axes).expand(-1, *([window_size] * axes), -1) + mesh
    return window % grid_size


def northity(input, radius=0.5, axis_dim=None, one_hot=False):
    if axis_dim is None:
        axis_dim = input.shape[:-1]

    needle = input.view(*input.shape[:-1], -1, axis_dim)

    north = F.one_hot(torch.tensor(axis_dim - 1), axis_dim) if one_hot \
        else torch.ones(axis_dim)

    return F.cosine_similarity(needle, north, -1) * radius


def compass_wheel_localize(input, num_degrees=10, axis_dim=None):
    nrt = northity(input, num_degrees / 2, axis_dim) + num_degrees / 2
    quant = torch.round(nrt)
    return (nrt - (nrt - quant).detach()).long()

G = torch.randn([30, 10, 1])  # (N x ...) n times x V1 x ...
# print("grid:")
# print(G)
# Sample
P = torch.rand((2, 2 * 2)) * 2 - 1  # B x 2 * n
# print("pos:")
# print(P)
P = compass_wheel_localize(P, 10, 2)
# print("quant:")
# print(P)
W = coord_to_window(P, 3, 10)  # B x (M x ...) n times x n

t = time.time()
mem = lookup(G, W)  # B x (M x ...) n times x V1 x ...
print('read time', time.time() - t)

# print('mem:')
# print(mem.shape)
# print(mem)

# optim = Adam([G], lr=0.8)

# mem.requires_grad = True

# clone_G = G.clone()
# clone_mem = mem.clone()

# loss = mem.mean()
# t - time.time()
# loss.backward()
# optim.step()
# print('back time', time.time() - t)

clone_G = G.clone()

# print(W.shape)
# W, inds, counts = torch.unique(W, dim=0, return_inverse=True, return_counts=True)
# print(W.shape, counts)

t - time.time()
write(G, W, mem.uniform_())
print('write time', time.time() - t)
#
assert (G == clone_G).all()
# assert (mem == clone_mem).all()

