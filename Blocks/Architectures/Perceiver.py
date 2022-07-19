# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

from torch import nn

import Utils
from Blocks.Architectures import MLP
from Blocks.Architectures.Transformer import CrossAttentionBlock, PositionalEncodings, \
    LearnableFourierPositionalEncodings


class Perceiver(nn.Module):
    """Perceiver (https://arxiv.org/abs/2103.03206)
    Generalized to arbitrary spatial dimensions, dimensionality-agnostic I/O w.r.t. state dict.
    For consistency with Vision models, assumes channels-first!"""
    def __init__(self, input_shape=(32,), num_tokens=32, num_heads=None, token_dim=None, output_dim=None,
                 depths=None, recursions=None, learnable_tokens=True, channels_first=True,
                 learnable_positional_encodings=False, positional_encodings=True):
        super().__init__()

        positional_encodings = LearnableFourierPositionalEncodings if learnable_positional_encodings \
            else PositionalEncodings if positional_encodings else nn.Identity

        self.positional_encodings = positional_encodings(input_shape, channels_first=channels_first)

        # Dimensions

        self.channels_first = channels_first

        shape = Utils.cnn_feature_shape(input_shape, self.positional_encodings)

        self.input_dim = shape if isinstance(shape, int) else shape[0] if channels_first else shape[-1]
        self.token_dim = token_dim or self.input_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        depths = [3] if depths is None else depths
        recursions = [1 for _ in depths] if recursions is None else recursions

        assert len(depths) == len(recursions), f'Recursion must be specified for each depth: {recursions}, {depths}'
        assert self.token_dim == self.input_dim or recursions[0] == 1, \
            f'First depth cannot be recursive if token_dim ≠ input_dim, {recursions}, {self.token_dim}≠{self.input_dim}'

        # Input tokens

        tokens = torch.zeros(1, self.token_dim, self.num_tokens) if self.channels_first \
            else torch.zeros(1, self.num_tokens, self.token_dim)

        self.tokens = PositionalEncodings(tokens.shape[1:], 0, channels_first=channels_first)(tokens)

        if learnable_tokens:
            self.tokens = nn.Parameter(self.tokens)

        # Perceiver attention layers

        self.cross_attention = nn.ModuleList(sum([[CrossAttentionBlock(self.token_dim if i == 0 else self.input_dim,
                                                                       num_heads, self.input_dim,
                                                                       channels_first=channels_first)] * recurs
                                                  for i, recurs in enumerate(recursions)], []))

        self.self_attentions = nn.ModuleList(sum([[nn.Sequential(*[CrossAttentionBlock(self.input_dim, num_heads,
                                                                                       channels_first=channels_first)
                                                                   for _ in range(depth - 1)])] * recurs
                                                  for recurs, depth in zip(recursions, depths)], []))

        # Output tokens

        if self.output_dim is not None:
            outputs = torch.zeros(1, self.token_dim, self.output_dim) if self.channels_first \
                else torch.zeros(1, self.output_dim, self.token_dim)

            self.outputs = PositionalEncodings(outputs.shape[1:], 0, channels_first=channels_first)(outputs)

            if learnable_tokens:
                self.outputs = nn.Parameter(self.outputs)

            self.output_attention = CrossAttentionBlock(self.output_dim, num_heads, self.token_dim,
                                                        channels_first=channels_first)

            self.MLP = MLP(self.token_dim, 1, self.token_dim, activation=nn.GELU())

    def repr_shape(self, *_):
        # Passed-in output dim, or same shape as tokens
        return (self.output_dim,) if self.output_dim else (self.token_dim, self.num_tokens) if self.channels_first \
            else (self.num_tokens, self.token_dim)

    def forward(self, input):
        input = self.positional_encodings(input)
        output = self.tokens

        for cross_attention, self_attentions in zip(self.cross_attention, self.self_attentions):
            output = self_attentions(cross_attention(output, input))

        if self.output_dim:
            output = self.output_attention(self.outputs, output)

            if self.channels_first:
                output = Utils.ChSwap(output)

            output = self.MLP(output).squeeze(-1)

        return output


# With dynamic reshaping draft
# class Perceiver(nn.Module):
#     """Perceiver (https://arxiv.org/abs/2103.03206)
#     Generalized to arbitrary spatial dimensions, dimensionality-agnostic I/O w.r.t. state dict.
#     For consistency with Vision models, assumes channels-first!"""
#     def __init__(self, input_shape=(32,), num_tokens=32, num_heads=None, token_dim=None, output_dim=None,
#                  depths=None, recursions=None, learnable_tokens=True, channels_first=True,
#                  learnable_positional_encodings=False, positional_encodings=True):
#         super().__init__()
#
#         positional_encodings = LearnableFourierPositionalEncodings if learnable_positional_encodings \
#             else PositionalEncodings if positional_encodings else nn.Identity
#
#         self.positional_encodings = positional_encodings(input_shape, channels_first=channels_first)
#
#         # Dimensions
#
#         self.channels_first = channels_first
#
#         shape = Utils.cnn_feature_shape(input_shape, positional_encodings)
#
#         self.input_dim = shape if isinstance(shape, int) else shape[0] if channels_first else shape[-1]
#         self.token_dim = token_dim or self.input_dim
#         self.output_dim = output_dim
#         self.num_tokens = num_tokens
#
#         depths = [3] if depths is None else depths
#         recursions = [1 for _ in depths] if recursions is None else recursions
#
#         assert len(depths) == len(recursions), f'Recursion must be specified for each depth: {recursions}, {depths}'
#         assert self.token_dim == self.input_dim or recursions[0] == 1, \
#             'First depth cannot be recursive if token_dim ≠ input_dim'
#
#         # Input tokens
#
#         tokens = torch.zeros(1, self.token_dim, self.num_tokens) if self.channels_first \
#             else torch.zeros(1, self.num_tokens, self.token_dim)
#
#         self.tokens = PositionalEncodings(tokens.shape[1:], 0, channels_first=channels_first)(tokens)
#
#         if learnable_tokens:
#             self.tokens = nn.Parameter(self.tokens)
#
#         # Perceiver attention layers
#
#         self.cross_attention = nn.ModuleList(sum([[CrossAttentionBlock(self.token_dim if i == 0 else self.input_dim,
#                                                                        num_heads, self.input_dim,
#                                                                        channels_first=channels_first)] * recurs
#                                                   for i, recurs in enumerate(recursions)], []))
#
#         self.self_attentions = nn.ModuleList(sum([[nn.Sequential(*[CrossAttentionBlock(self.input_dim, num_heads,
#                                                                                        channels_first=channels_first)
#                                                                    for _ in range(depth - 1)])] * recurs
#                                                   for recurs, depth in zip(recursions, depths)], []))
#
#         # Output tokens
#
#         if self.output_dim is not None:
#             outputs = torch.zeros(1, self.output_dim, self.token_dim)
#
#             self.outputs = PositionalEncodings(outputs.shape[1:], 0, channels_first=False)(outputs)
#
#             self.output_attention = CrossAttentionBlock(self.output_dim, num_heads, self.token_dim,
#                                                         channels_first=channels_first)
#
#             self.MLP = MLP(self.token_dim, 1, self.token_dim, activation=nn.GELU())
#
#             if learnable_tokens:
#                 self.outputs = nn.Parameter(self.outputs)
#
#     def repr_shape(self, *_):
#         return (self.output_dim,) if self.output_dim else (self.token_dim, self.num_tokens) if self.channels_first \
#             else (self.num_tokens, self.token_dim)
#
#     def forward(self, input, output_dim=None):
#         input = self.positional_encodings(input)
#         output = self.tokens
#
#         for cross_attention, self_attentions in zip(self.cross_attention, self.self_attentions):
#             output = self_attentions(cross_attention(output, input))
#
#         if self.output_dim:
#             self.outputs = self.outputs[:, :output_dim, :]  # Dynamic reshaping possible
#
#             if self.channels_first:
#                 output = Utils.ChSwap(self.outputs, False)
#
#             output = self.output_attention(self.outputs, output)
#
#             if self.channels_first:
#                 output = Utils.ChSwap(output, False)
#
#             output = self.MLP(output).squeeze(-1)
#
#         return output
