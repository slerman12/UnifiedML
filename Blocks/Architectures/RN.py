# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch
from torch import nn

import Utils
from Blocks.Architectures import MLP
from Blocks.Architectures.Transformer import PositionalEncodings, LearnableFourierPositionalEncodings


class RN(nn.Module):
    """Relation Network (https://arxiv.org/abs/1706.01427), generalized to arbitrary spatial dims.
    Supports positional encodings. For consistency with Vision models, assumes channels-first!"""
    def __init__(self, input_shape=(32,), context_dim=None, inner_depth=3, outer_depth=2, hidden_dim=None,
                 output_dim=None, mid_nonlinearity=nn.Identity(), dropout=0, channels_first=True,
                 learnable_positional_encodings=False, positional_encodings=True):
        super().__init__()

        positional_encodings = LearnableFourierPositionalEncodings if learnable_positional_encodings \
            else PositionalEncodings if positional_encodings else nn.Identity

        self.positional_encodings = positional_encodings(input_shape, channels_first=channels_first)

        # Dimensions

        self.channels_first = channels_first

        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        shape = Utils.cnn_feature_shape(input_shape, self.positional_encodings)
        self.input_dim = shape[0] if channels_first else shape[-1]

        # Defaults
        self.context_dim = context_dim or self.input_dim
        self.hidden_dim = hidden_dim or self.input_dim * 4
        self.output_dim = output_dim or self.input_dim

        self.inner = nn.Sequential(
            MLP(self.input_dim + self.context_dim, self.hidden_dim, self.hidden_dim, inner_depth), nn.Dropout(dropout))
        self.mid_nonlinearity = mid_nonlinearity
        self.outer = MLP(self.hidden_dim, self.output_dim, self.hidden_dim, outer_depth)

    def repr_shape(self, *_):
        return self.output_dim,

    def forward(self, input, context=None):
        input = self.positional_encodings(input)

        if context is None:
            context = input  # Self-attention

        # Permute as channels-last
        if self.channels_first:
            input, context = [Utils.ChSwap(x, False) for x in [input, context]]

        # Flatten intermediary spatial dims
        input = input.flatten(1, -2)
        context = context.flatten(1, -2)

        # Validate shapes
        assert input.shape[-1] == self.input_dim, f'Unexpected input shape {input.shape[-1]}≠{self.input_dim}'
        assert context.shape[-1] == self.context_dim, f'Unexpected context shape {context.shape[-1]}≠{self.context_dim}'

        input = input.unsqueeze(1).expand(-1, context.shape[1], -1, -1)
        context = context.unsqueeze(2).expand(-1, -1, input.shape[2], -1)
        pairs = torch.cat([input, context], -1)

        relations = self.inner(pairs)
        mid = self.mid_nonlinearity(relations.sum(1).sum(1))
        outer = self.outer(mid)

        try:
            output = outer.view(len(input), self.output_dim)  # Validate shape
        except RuntimeError:
            raise RuntimeError(f'\nUnexpected output shape {tuple(output.shape)}, ≠{(len(input), self.output_dim)}')

        # Convert to channels-first
        if self.channels_first:
            output = Utils.ChSwap(output, False)

        return output
