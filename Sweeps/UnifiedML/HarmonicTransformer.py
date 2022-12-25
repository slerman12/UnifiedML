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
    # ['1D CNN', '"Harmonic Transformer"'],  # TODO
    # TODO or, with title 'Dimensionality Adaptivity: [1D] CNN And ["]Harmonic Transformer["] On 1D Audio'
    # ['2D CNN', 'Vanilla ViT'],
]

runs.UnifiedML.bluehive = False
runs.UnifiedML.lab = True
runs.UnifiedML.title = 'Dimensionality Adaptivity: 2D-CNN And Vanilla ViT On 1D Audio'
# runs.UnifiedML.x_axis = 'time'


# # TODO Maybe a Custom Datasets version with spectrogram and CNN, or for TinyImageNet?
