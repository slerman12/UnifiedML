import torch


# Given N1 x ... x Nn x X1 x ... x Xm grid and window B x M1 x ... x Mn returns tensor B x M1 x ... x Mn x X1 x ... x Xm

# N x ... x N x X
N = 5
X = 3

# B x M x ... x M
B = 8
M = 2

grid = torch.arange(N * N * X).view(N, N, X)
window = torch.randint(0, N, (B, M, M))

# Naive (slow) solution:
g = grid.expand(B, N, N, X)
w = window.unsqueeze(-1).expand(B, M, M, X)
out = torch.gather(g, 1, w)

print(out.size())
assert torch.Size((B, M, M, X)) == out.size()




# (Multi-dim) indexing, assumes a single index is 1D
def gather_indices(item, ind, dim=-1):
    assert item.shape[-len(ind.shape):dim] == \
           ind.shape[-min(len(item[:dim].shape) + 1,
                          len(ind.shape)):-1], "Can't broadcast index to item"
    diff = max(0, len(ind.shape) - len(item[:dim].shape) + 1)
    ind = ind.long().view(*ind.shape[:diff], *[1 for _ in item.shape[:dim]], ind.shape[-1])
    ind = ind.long().expand(*ind.shape[:diff], *item.shape[:dim], ind.shape[-1])
    if -1 < dim < len(item.shape) - 1:
        trail_shape = item.shape[dim + 1:]
        ind = ind.reshape(ind.shape + (1,)*len(trail_shape))
        ind = ind.expand(*ind.shape[:dim + 1], *trail_shape)
    item = item.view(*ind[:-1].shape, *item[dim:].shape)
    dim = dim + diff if dim >= 0 else dim
    out = torch.gather(item, dim, ind)
    for _ in range(diff):
        out = out.squeeze(0)
    return out


def gather_indices_Nd(item, ind, dim=-1):
    pass