import torch
from torch import nn
from torch import einsum

import copy
from einops import rearrange

from Blocks.Architectures.MLP import MLP

import Utils


class CrossAttention(nn.Module):
    def __init__(self, dim=32, heads=8, context_dim=None):
        super().__init__()

        assert heads % dim == 0
        self.dim = dim
        self.heads = heads

        context_dim = dim if context_dim is None \
            else context_dim

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim * 2, bias=False)

    def forward(self, x, context):
        # Conserves shape
        shape = x.shape
        assert shape[-1] == self.dim

        x = x.flatten(1, -2)
        context = context.flatten(1, -2)

        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.dim ** -0.5

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)

        # Restores original shape
        return out.view(shape)


class SelfAttention(CrossAttention):
    def forward(self, x, *args):
        return super()(x, x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=32, heads=8, context_dim=None, optim_lr=None, target_tau=None):
        super().__init__()

        self.dim = dim
        self.heads = heads

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, heads, context_dim)
        self.mlp = MLP(dim, dim, dim, 2, nn.GELU())

        self.init(optim_lr, target_tau)

    def init(self, optim_lr=None, target_tau=None):
        # Initialize weights
        self.apply(Utils.weight_init)

        # Optimizer
        if optim_lr is not None:
            self.optim = torch.optim.Adam(self.parameters(), lr=optim_lr)

        # EMA
        if target_tau is not None:
            self.target = copy.deepcopy(self)
            self.target_tau = target_tau

    def forward(self, x, context):
        attn = self.ln1(self.attn(x, context)) + x
        out = self.ln2(self.mlp(attn)) + attn

        return out


class SelfAttentionBlock(CrossAttentionBlock):
    def forward(self, x, *_):
        return super()(x, x)
