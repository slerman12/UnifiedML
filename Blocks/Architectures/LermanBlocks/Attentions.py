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
    Can be useful as a lightweight perceiver, although for subsequent layers can just use fixed weights and ReLU,
    like cortical columns, an Ultra-Lightweight Perceiver; separates positional encodings from computation/reasoning

    Among my BioNets? e.g. BioPerceiver

    More Bio-Plausible, instead of pos, have a BioNeuron encoding span time (reference frame)

    But cross attention from pos to input not bio-plausible?"""
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


class LermanBot(nn.Module):
    def __init__(
            self,
            num_freq_bands,
            max_freq,
            freq_base=2,
            x_channels=3,
            x_axis=1,
            depth=2,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            cross_head_dim=64,
            latent_heads=8,
            latent_head_dim=64,
            self_per_cross_attn=3,
            weight_tie_layers=False,
            attn_dropout=0,
            ff_dropout=0,
    ):
        super().__init__()
        self.x_axis = 1 if not x_axis else x_axis
        self.num_freq_bands = num_freq_bands
        self.max_freq = max_freq
        self.freq_base = freq_base

        self.latent_dim = latent_dim

        x_fourier_channels = (x_axis * ((num_freq_bands * 2) + 1))
        # x_dim = x_fourier_channels + x_channels
        y_fourier_channels = ((num_freq_bands * 2) + 1)
        # y_dim = y_fourier_channels + 1

        def attend_pos(pos_dim, input_dim, latents):
            return Sequential(PosAttention(pos_dim, input_dim, latents, latent_dim, attn_dropout),
                              Residual(FeedForward(latent_dim, dropout=ff_dropout)))

        def attend_tb(dim, num_b, k, norm_query=True):
            norm_query_dim = dim if norm_query else None
            return Sequential(Residual(PreNorm(norm_query_dim, TBAttention(dim, num_b, k, latent_heads, latent_head_dim,
                                                                           attn_dropout))),
                              Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout))))

        def attend_tb_bio(dim, num_b, k, norm_query=True):
            norm_query_dim = dim if norm_query else None
            return Sequential(Residual(PreNorm(norm_query_dim, TBAttentionBio(dim, num_b, k, latent_heads,
                                                                              latent_head_dim, attn_dropout))),
                              Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout))))

        def attend(query_dim, context_dim, cross=False, norm_query=True, norm_context=False):
            heads = cross_heads if cross else latent_heads
            head_dim = cross_head_dim if cross else latent_head_dim
            norm_query_dim = query_dim if norm_query else None
            norm_context_dim = context_dim if norm_context else None
            return Sequential(Residual(PreNorm(norm_query_dim, Attention(query_dim, context_dim, heads, head_dim,
                                                                         attn_dropout), norm_context_dim)),
                              Residual(PreNorm(query_dim, FeedForward(query_dim, dropout=ff_dropout))))

        self.enc_pos_attn_x = attend_pos(x_fourier_channels, x_channels, num_latents)
        self.enc_pos_attn_y = attend_pos(y_fourier_channels, 1, num_latents)
        # self.in_attn = attend(latent_dim, latent_dim)

        self.psi_attn = attend_tb_bio(latent_dim, num_b=1000, k=num_latents)

        get_latent_attn = cache_fn(lambda: attend(latent_dim, latent_dim))
        get_self_attn, get_tb_attn = map(cache_fn, (lambda: attend(latent_dim, latent_dim),
                                                    lambda: attend_tb_bio(latent_dim, num_b=1000, k=num_latents)))

        self.encoder = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            # should_cache = weight_tie_layers  # todo
            cache_args = {'cache': should_cache, 'cache_id': 0}

            self_attns = []

            for _ in range(self_per_cross_attn):
                self_attns.append(get_self_attn(**cache_args))

            self.encoder.append(nn.ModuleList([get_tb_attn(**cache_args), nn.Sequential(*self_attns)]))

        self.cross_attn_y_pred = attend(y_fourier_channels, latent_dim,
                                        cross=True,  # more heads?
                                        # residual=False,  # todo?
                                        norm_query=True, norm_context=True)

        self.to_y_out = nn.Linear(y_fourier_channels, 1)

    def forward(
            self,
            x,
            y_dim=1,
            psi=None,
            # x_sup=None,
            y_prev=None,
            truncate=False,
            mask=None,
    ):
        # encode x
        b, *x_axis_shape, _, device = *x.shape, x.device
        if len(x_axis_shape) == 0:
            x = x.unsqueeze(2)
            x_axis_shape = [x.shape[1]]
        assert len(x_axis_shape) == self.x_axis, 'input data must have the right number of axis'
        if truncate:
            x = x.detach()
        x = rearrange(x, 'b ... d -> b (...) d')
        # todo initialize as x_pos or zeros, then update with backwards residuals
        x_pos = fourier_pos(batch_size=b, axis=x_axis_shape, max_freq=self.max_freq, num_freq_bands=self.num_freq_bands,
                            freq_base=self.freq_base, device=device)
        x_pos = rearrange(x_pos, 'b ... d -> b (...) d')
        enc = self.enc_pos_attn_x(x_pos, x, mask=mask)

        # encode y
        # todo can update with backwards residuals as well
        y_pos = fourier_pos(batch_size=b, axis=[y_dim], max_freq=self.max_freq, num_freq_bands=self.num_freq_bands,
                            freq_base=self.freq_base, device=device)
        if exists(y_prev):
            assert y_prev.shape[0] == b and len(y_prev.shape) == 2
            if truncate:
                y_prev = y_prev.detach()
            y_prev = y_prev.unsqueeze(2)
            enc_y = self.enc_pos_attn_y(y_pos, y_prev, mask=mask)
            enc = enc + enc_y

        # enc = self.in_attn(enc)

        # memory
        if exists(psi):
            if truncate:
                psi = [p.detach() for p in psi]
        psi = default(psi, [None] * len(self.encoder))
        # enc, psi = self.psi_attn(enc, psi=psi)

        # layers
        for i, (tb_attn, self_attns) in enumerate(self.encoder):
            enc, psi[i] = tb_attn(enc, psi=psi[i])
            enc = self_attns(enc)

        # decoder
        y_pred_latents = self.cross_attn_y_pred(y_pos, context=enc, mask=mask)
        y_pred = self.to_y_out(y_pred_latents).squeeze(-1)
        return y_pred, psi