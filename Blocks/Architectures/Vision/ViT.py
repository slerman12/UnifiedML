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
    def __init__(self, input_shape, patch_size=4, dim=32, heads=8, depth=3, num_classes=1000, pool='cls'):
        super().__init__()

        in_channels = input_shape[0]
        image_size = input_shape[1]

        assert input_shape[1] == input_shape[2], 'Compatible with square images only'
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.attn = nn.Sequential(*[SelfAttentionBlock(dim, heads) for _ in range(depth)])

        # self.pool = pool
        #
        # self.repr = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = self.attn(x)

        return x
        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        #
        # return self.repr(x)
