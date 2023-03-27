# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Sweeps.Templates import template


runs = template('UnifiedML')

runs.UnifiedML.sweep = [
    f"""
    task=classify/celeba
    Dataset=Datasets.Suites._CelebA.CelebA 
    +dataset.username=slerman
    +dataset.key=a55142727f5fbd5029de3e7597902ff9
    generate=true 
    Discriminator=DCGAN.Discriminator 
    Generator=DCGAN.Generator 
    experiment=DCGAN
    train_steps=10000
    time="5-00:00:00"
    lab=true
    autocast=false
    mem=160
    """,  # Note: Manually set "pseudonym" to task_name in sbatch.yaml
]


runs.UnifiedML.plots = [
    ['DCGAN'], ['DCGAN2']
]

runs.UnifiedML.sftp = True
runs.UnifiedML.bluehive = False
runs.UnifiedML.lab = True
runs.UnifiedML.title = 'DCGAN'
