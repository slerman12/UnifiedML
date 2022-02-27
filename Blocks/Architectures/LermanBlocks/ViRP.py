# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Blocks.Architectures.LermanBlocks.MultiHeadRelation import *
from Blocks.Architectures.MultiHeadAttention import CrossAttentionBlock


class ViRP(ViT):
    def __init__(self, input_shape, patch_size=4, out_channels=32, heads=8, depth=3, pool='cls', output_dim=None,
                 experiment='head_head_in_RN'):
        super().__init__(input_shape, patch_size, out_channels, heads, depth, pool, True, output_dim)

        if experiment == 'concat_plus_mid':
            core = RelationSimplest  # not important
        elif experiment == 'concat_plus_in':  # velocity reasoning from mlp only
            core = RelationSimplestV2
        elif experiment == 'plus_in_concat_plus_mid':  # see if attention is useful as "reason-er"
            core = RelationSimplestV3
        elif experiment == 'head_wise_ln':  # disentangled relational reasoning - are the heads independent?
            core = RelationDisentangled
        elif experiment == 'head_in_RN':  # invariant relational reasoning between input-head - are they?
            core = RelationSimpler
        elif experiment == 'head_head_in_RN':  # relational reasoning between heads
            core = RelationRelative
        elif experiment == 'head_head_RN_plus_in':  # does reason-er only need heads independent of input/tokens?
            core = RelationSimplerV3
        else:
            # layernorm values, confidence
            # see if more mhdpa layers picks up the load - is the model capacity equalized when layers are compounded?
            core = RelationRelative

        self.attn = nn.Sequential(*[core(out_channels, heads) for _ in range(depth)])

