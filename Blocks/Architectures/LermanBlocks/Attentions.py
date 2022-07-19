import math
from math import pi, log
from functools import wraps
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import init, Parameter


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
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


class PosAttention(nn.Module):
    """Weighs the x solely based on their positions! This is how MLP weights weigh anyway + a non-linearity
    Can be useful as a lightweight perceiver, although for subsequent layers can just use fixed weights,
    like cortical columns, an Ultra-Lightweight Perceiver; separates positional encodings from computation/reasoning

    Among my BioNets? e.g. BioPerceiver"""
    def __init__(self, pos_dim, x_dim, num_latents, latent_dim=64, dropout=0):
        super().__init__()
        num_heads = out_dim = latent_dim
        inner_dim = latent_dim * num_heads
        self.scale = latent_dim ** -0.5
        self.num_heads = num_heads
        self.x_dim = x_dim

        self.to_q = nn.Linear(pos_dim, inner_dim, bias=False)
        # might want mult heads too
        self.k = nn.Parameter(torch.randn(num_latents, latent_dim))
        # if project_x:
        #     self.to_v = nn.Linear(x_dim, inner_dim, bias=False)

        # self.to_out = nn.Sequential(
        #     nn.Linear(x_dim * num_heads, out_dim),
        #     nn.Dropout(dropout)
        # )
        self.dropout = nn.Dropout(dropout)  # TODO added this after repo update
        self.to_out = nn.Linear(x_dim * num_heads, out_dim)

    def forward(self, pos, x, mask=None):
        h = self.num_heads

        q = self.to_q(pos)

        q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)

        if self.x_dim == 1 and x.shape[-1] != 1:
            x = x.unsqueeze(-1)
        assert x.shape[-1] == self.x_dim
        v = x  # b n 1

        sim = einsum('b n d, j d -> b j n', q, self.k) * self.scale

        # if exists(mask):
        #     mask = rearrange(mask, 'b ... -> b (...)')
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, 'b j -> (b h) () j', h=h)
        #     sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)  # TODO added this after repo update
        attn = rearrange(attn, '(b h) n d -> b h n d', h=h)

        out = einsum('b h j n, b n d -> b h j d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class TBAttention(nn.Module):
    """Thousand Brains Attention
    Uses query to select top k "brains" (each with corresponding key), then applies brains on values
    and weighs by similarity. These guys unknowingly copied me: https://arxiv.org/abs/2110.06399"""
    def __init__(self, dim, num_b=1000, top_k=1000, heads=8, head_dim=64, dropout=0):
        super().__init__()
        inner_dim = head_dim * heads
        self.scale = head_dim ** -0.5
        self.num_heads = heads

        self.num_b = num_b
        self.top_k = top_k

        self.b = Parameter(torch.Tensor(num_b, head_dim, head_dim))  # todo add bias
        self.k = Parameter(torch.Tensor(num_b, head_dim))
        self.to_qv = nn.Linear(dim, inner_dim * 2, bias=False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # )
        self.dropout = nn.Dropout(dropout)  # TODO added this after repo update
        self.to_out = nn.Linear(inner_dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.b, a=math.sqrt(5))
        init.kaiming_uniform_(self.k, a=math.sqrt(5))

    def forward(self, x, mask=None):
        h = self.num_heads

        q, v = self.to_qv(x).chunk(2, dim=-1)

        q, v = map(lambda t: rearrange(t, 'b i (h d) -> (b h) i d', h=h), (q, v))

        sim = einsum('n d, b i d -> b i n', self.k, q) * self.scale  # scaled importances

        # if exists(mask):
        #     mask = rearrange(mask, 'b ... -> b (...)')
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, 'b j -> (b h) () j', h=h)
        #     sim.masked_fill_(~mask, max_neg_value)

        t, b_inds = torch.topk(sim, self.top_k)  # b i k
        b = self.b[b_inds]  # b i k d d
        attn = t.softmax(dim=-1)
        attn = self.dropout(attn)  # TODO added this after repo update

        out = einsum('b i k d d, b i d -> b i k d', b, v)
        out = einsum('b i k d, b i k -> b i d', out, attn)
        out = rearrange(out, '(b h) i d -> b i (h d)', h=h)
        return self.to_out(out)


class TBAttentionBio(nn.Module):
    def __init__(self, dim, num_b=1000, top_k=1000, heads=8, head_dim=64, dropout=0):
        super().__init__()
        inner_dim = head_dim * heads
        self.scale = head_dim ** -0.5
        self.num_heads = heads

        self.num_b = num_b
        self.top_k = top_k

        self.b = Parameter(torch.Tensor(num_b, head_dim, head_dim))  # todo add bias
        self.k = Parameter(torch.Tensor(num_b, head_dim))
        self.o = Parameter(torch.Tensor(num_b, head_dim))
        self.to_qv = nn.Linear(dim, inner_dim * 2, bias=False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # )
        self.dropout = nn.Dropout(dropout)  # TODO added this after repo update
        self.to_out = nn.Linear(inner_dim, dim)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.b, a=math.sqrt(5))
        init.kaiming_uniform_(self.k, a=math.sqrt(5))
        init.kaiming_uniform_(self.o, a=math.sqrt(5))

    def forward(self, x, psi=None, mask=None):
        b = x.shape[0]
        h = self.num_heads
        i = x.shape[1]
        n = self.num_b

        q, v = self.to_qv(x).chunk(2, dim=-1)

        q, v = map(lambda t: rearrange(t, 'b i (h d) -> (b h) i d', h=h), (q, v))

        d = q.shape[-1]

        sim = einsum('n d, b i d -> b i n', self.k, q) * self.scale  # scaled importances

        # if exists(mask):
        #     mask = rearrange(mask, 'b ... -> b (...)')
        #     max_neg_value = -torch.finfo(sim.dtype).max
        #     mask = repeat(mask, 'b j -> (b h) () j', h=h)
        #     sim.masked_fill_(~mask, max_neg_value)

        t, b_inds = torch.topk(sim, self.top_k)  # b i k
        brains = self.b[b_inds]  # b i k d d
        attn = t.softmax(dim=-1)
        attn = self.dropout(attn)  # TODO added this after repo update

        diff = einsum('b i k d d, b i d -> b i k d', brains, v)
        if psi is not None:
            print(psi.shape, b_inds.shape)
        membr_pot = diff if psi is None else psi[b_inds] + diff
        # if psi is not None:
        #     print(psi[:, :, b_inds].shape, diff.shape)
        spike_proba = torch.sigmoid(membr_pot)
        spike = spike_proba + (spike_proba.round() - spike_proba).detach()
        nrtrnsmtr = spike * self.o[b_inds]
        psi = default(psi, torch.zeros_like(self.o))  # n d
        # psi.scatter_(2, b_inds, (1 - spike) * membr_pot)
        # print(psi.shape, b_inds.shape,
        #       psi[b_inds].shape,
        #       ((1 - spike_proba) * membr_pot).shape)
        psi[b_inds] = (1 - spike_proba) * membr_pot
        print(psi.shape)
        # psi = (1 - spike_proba) * membr_pot

        out = einsum('b i k d, b i k -> b i d', nrtrnsmtr, attn)
        out = rearrange(out, '(b h) i d -> b i (h d)', h=h)
        return self.to_out(out), psi
