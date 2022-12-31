# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Sweeps.Templates import convert_to_attr_dict


"""
Structure of runs:

-> Sweep Group:
        -> Sweep & Plots & Plots Metadata
"""


from Sweeps.Templates import template


runs = template('Example')

# runs.Example.sweep = [
#     'experiment=Test1 task=classify/mnist train_steps=2000'
# ]
#
runs.Example.plots = [
    ['.*Test1'],
]
# runs.Example.title = 'A Good Ol\' Thing'

runs.Example.bluehive = False


# from Sweeps.XRD.XRDPaper import runs
# from Sweeps.XRD.LATT import runs
# from Sweeps.UnifiedML.ClassifyRL import runs
# from Sweeps.UnifiedML.Atari import runs
# from Sweeps.UnifiedML.Mario import runs
# from Sweeps.UnifiedML.Atari import runs
# from Sweeps.UnifiedML.DMC import runs
# from Sweeps.UnifiedML.HarmonicTransformer import runs
# from Sweeps.UnifiedML.ClassifyRL2 import runs
from Sweeps.UnifiedML.DuelingDQN import runs


runs = convert_to_attr_dict(runs)  # Necessary if runs is defined as a dict!
