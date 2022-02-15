# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch import nn

from einops import repeat
from einops.layers.torch import Rearrange

from Blocks.Architectures.MultiHeadAttention import SelfAttentionBlock


class ViT(nn.Module):
    def __init__(self, input_shape, patch_size=4, out_channels=32, heads=8, depth=3, pool='cls', output_dim=None):
        super().__init__()

        in_channels = input_shape[0]
        image_size = input_shape[1]
        self.patch_size = patch_size
        self.output_dim = output_dim

        assert input_shape[1] == input_shape[2], 'Compatible with square images only'
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, out_channels),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, out_channels))
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channels))

        h, w = self.feature_shape(*input_shape[1:])

        self.attn = nn.Sequential(*[SelfAttentionBlock(out_channels, heads) for _ in range(depth)],
                                  Rearrange('b (h w) c -> b c h w', h=h, w=w)  # Channels 1st
                                  )

        if output_dim is not None:
            self.pool = pool

            self.repr = nn.Sequential(
                nn.LayerNorm(out_channels),
                nn.Linear(out_channels, output_dim)
            )

    def feature_shape(self, h, w):
        return 1, (h // self.patch_size) * (w // self.patch_size) + 1

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = self.attn(x)

        if self.output_dim is not None:
            x = x.transpose(-1, -3)  # Channels last
            x = x.flatten(1, -2)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            return self.repr(x)
        return x
