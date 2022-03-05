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

        # self.tokens = nn.Parameter(torch.randn(tokens, token_dim))
        self.tokens = torch.randn(tokens, token_dim)
        init.kaiming_uniform_(self.tokens, a=math.sqrt(5))

        self.attn_token = CrossAttentionBlock(token_dim, heads, dim, value_dim, relu=relu)
        self.reattn_token = CrossAttentionBlock(value_dim, heads, dim, value_dim, relu=relu)
        self.attn = nn.Sequential(*[CrossAttentionBlock(value_dim, heads, relu=relu) for _ in range(depth - 1)])

    def forward(self, x):
        tokens = self.attn_token(self.tokens, x)
        x = self.reattn_token(tokens, x)
        return self.attn(x)


class PerceiverV2(nn.Module):
    def __init__(self, dim, heads, tokens=32, token_dim=None, value_dim=None, depths=None, recursions=None, relu=False):
        super().__init__()

        token_dim = dim if token_dim is None else token_dim
        value_dim = dim if value_dim is None else value_dim

        depths = [3] if depths is None else depths
        recursions = [1 for _ in depths] if recursions is None else recursions

        assert len(depths) == len(recursions), 'Recursion must be specified for each depth'
        assert token_dim == value_dim or recursions[0] == 1, 'First depth cannot be recursive if token_dim ≠ value_dim'

        self.tokens = nn.Parameter(torch.randn(tokens, token_dim))
        init.kaiming_uniform_(self.tokens, a=math.sqrt(5))

        self.reattn = nn.ModuleList(sum([[CrossAttentionBlock(token_dim if i == 0 else value_dim,
                                                              heads, dim, value_dim, relu=relu)] * recurs
                                         for i, recurs in enumerate(recursions)], []))
        self.attn = nn.ModuleList(sum([[nn.Sequential(*[CrossAttentionBlock(value_dim, heads, relu=relu)
                                                        for _ in range(inner_depth - 1)])] * recurs
                                       for recurs, inner_depth in zip(recursions, depths)], []))

    def forward(self, x):
        out = self.tokens
        for reattn, attn in zip(self.reattn, self.attn):
            out = attn(reattn(out, x))
        return out
