# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Sweeps.Templates import template


runs = template('UnifiedML')

runs.UnifiedML.sweep = [
    # Vanilla Harmonic Transformer
    f"""
    task=classify/custom
    Dataset=Datasets.Suites._SpeechCommands.SpeechCommands
    Aug=Identity
    Eyes=ViT
    experiment='HarmonicTransformer'
    """,

    # 1D-CNN
    f"""
    task=classify/custom
    Dataset=Datasets.Suites._SpeechCommands.SpeechCommands
    Aug=Identity
    experiment='1D-CNN'
    """,
]


runs.UnifiedML.plots = [
    ['1D-CNN', 'HarmonicTransformer'],
]

runs.UnifiedML.bluehive = False
runs.UnifiedML.lab = True
runs.UnifiedML.title = 'Dimensionality Adaptivity In UnifiedML via 2D-CNN and Vanilla ViT On 1D Audio'
