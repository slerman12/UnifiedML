# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import torch

from torch import nn
from torch.nn import init

from Blocks.Architectures import MLP
from Blocks.Architectures.MultiHeadAttention import CrossAttentionBlock


class Perceiver(nn.Module):
    def __init__(self, dim, heads=8, tokens=32, token_dim=None, value_dim=None, depth=3, relu=False, mlp_depth=None):
        super().__init__()

        token_dim = dim if token_dim is None else token_dim
        value_dim = dim if value_dim is None else value_dim

        self.tokens = nn.Parameter(torch.randn(tokens, token_dim))
        init.kaiming_uniform_(self.tokens, a=math.sqrt(5))

        self.mlp = nn.Identity() if mlp_depth is None \
            else nn.Sequential(MLP(dim, dim, dim, mlp_depth), nn.ReLU(inplace=True))
        self.attn_token = CrossAttentionBlock(token_dim, heads, dim, value_dim, relu=relu)
        self.reattn_token = CrossAttentionBlock(value_dim, heads, dim, value_dim, relu=relu)
        self.attn = nn.Sequential(*[CrossAttentionBlock(value_dim, heads, relu=relu) for _ in range(depth - 1)])

    def forward(self, x):
        x = self.mlp(x)
        tokens = self.attn_token(self.tokens, x)
        x = self.reattn_token(tokens, x)
        return self.attn(x)
