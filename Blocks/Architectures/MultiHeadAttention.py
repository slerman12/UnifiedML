import torch
from torch import nn
from torch import einsum

from einops import repeat, rearrange

import Utils


def attend(query_dim, context_dim, cross=False, norm_query=True, norm_context=False):
    heads = cross_heads if cross else latent_heads
    head_dim = cross_head_dim if cross else latent_head_dim
    norm_query_dim = query_dim if norm_query else None
    norm_context_dim = context_dim if norm_context else None
    return Sequential(Residual(PreNorm(norm_query_dim, Attention(query_dim, context_dim, heads, head_dim,
                                                                 attn_dropout), norm_context_dim)),
                      Residual(PreNorm(query_dim, FeedForward(query_dim, dropout=ff_dropout))))


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads

        context_dim = Utils.default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        context = Utils.default(context, x)

        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class Sequential(nn.Sequential):
    def forward(self, *args, **kwargs):
        for module in self._modules.values():
            output = module(*args, **kwargs)
            args = [output]
        return output


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, **kwargs):
        if isinstance(x, tuple):
            assert len(x) == 2
            return self.net(x[0]), x[1]
        return self.net(x)