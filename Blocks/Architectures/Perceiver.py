# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import torch

from torch import nn
from torch.nn import init

from einops import rearrange, repeat

from Blocks.Architectures.MultiHeadAttention import CrossAttention


class TokenAttention(CrossAttention):
    def __init__(self, dim=32, heads=None, tokens=8, token_dim=None, value_dim=None, talk_h=False, relu=False):
        if token_dim is None:
            token_dim = dim

        super().__init__(token_dim, heads, dim, value_dim, talk_h, relu)

        self.tokens = nn.Parameter(torch.randn(tokens, token_dim))
        init.kaiming_uniform_(self.tokens, a=math.sqrt(5))

    def forward(self, x, *_):
        return super().forward(self.tokens, x)


class Perceiver(nn.Module):
    def __init__(
            self,
            *,
            fourier_encode_data=True,
            num_freq_bands,
            max_freq,
            freq_base=2,
            input_channels=3,
            input_axis=2,
            depth,
            logits_dim=None,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            attn_dropout=0,
            ff_dropout=0,
    ):
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.freq_base = freq_base

        self.fourier_encode_data = fourier_encode_data
        in_fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        dim = in_fourier_channels + input_channels
        out_fourier_channels = ((num_freq_bands * 2) + 1) if fourier_encode_data else 0

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        init.kaiming_uniform_(self.latents, a=math.sqrt(5))

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head, dropout=attn_dropout), context_dim=dim),
            PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        ])

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        # get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        # get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        # get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        # get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        #
        # get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))
        #
        # self.layers = nn.ModuleList([])
        # for i in range(depth):
        #     should_cache = i > 0 and weight_tie_layers
        #     cache_args = {'_cache': should_cache}
        #
        #     self_attns = nn.ModuleList([])
        #
        #     for _ in range(self_per_cross_attn):
        #         self_attns.append(nn.ModuleList([
        #             get_latent_attn(**cache_args),
        #             get_latent_ff(**cache_args)
        #         ]))
        #
        #     self.layers.append(nn.ModuleList([
        #         get_cross_attn(**cache_args),
        #         get_cross_ff(**cache_args),
        #         self_attns
        #     ]))
        #
        # self.to_logits = nn.Sequential(
        #     Reduce('b n d -> b d', 'mean'),
        #     nn.LayerNorm(latent_dim),
        #     nn.Linear(latent_dim, num_classes)
        # ) if final_classifier_head else nn.Identity()

        self.pos_encoder = FeedForward(out_fourier_channels, dropout=ff_dropout)

        self.decoder_cross_attn = PreNorm(out_fourier_channels, Attention(out_fourier_channels, latent_dim, heads=cross_heads, dim_head=cross_dim_head, dropout=attn_dropout), context_dim=latent_dim)
        self.decoder_ff = PreNorm(out_fourier_channels, FeedForward(out_fourier_channels, dropout=ff_dropout)) if decoder_ff else None

        self.to_logits = nn.Linear(out_fourier_channels, logits_dim) if exists(logits_dim) else nn.Identity()

    def forward(
            self,
            data,
            num_outputs=1,
            mask=None
    ):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base=self.freq_base)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)
            data = torch.cat((data, enc_pos), dim=-1)

        # concat to channels of data and flatten axis
        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b=b)

        cross_attn, cross_ff = self.cross_attend_blocks

        # cross attention only happens once for Perceiver IO
        x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x

        # layers
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # for cross_attn, cross_ff, self_attns in self.layers:
        #     x = cross_attn(x, context = data, mask = mask) + x
        #     x = cross_ff(x) + x
        #
        #     for self_attn, self_ff in self_attns:
        #         x = self_attn(x) + x
        #         x = self_ff(x) + x

        outs = None
        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), [num_outputs]))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands, base=self.freq_base)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            outs = repeat(enc_pos, '... -> b ...', b=b)
            outs = self.pos_encoder(outs)

        if not exists(outs):
            return x

        # cross attend from decoder queries to latents
        latents = self.decoder_cross_attn(outs, context=x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)

        # final linear out
        return self.to_logits(latents).squeeze(-1)
