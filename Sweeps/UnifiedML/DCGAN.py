# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Sweeps.Templates import template


runs = template('UnifiedML')

runs.UnifiedML.sweep = [
    f"""
    task=classify/tinyimagenet 
    generate=true 
    Discriminator=DCGAN.Discriminator 
    Generator=DCGAN.Generator 
    z_dim=100 
    'env.transform="transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64)])"' 
    experiment=DCGAN
    capacity=0
    time="5-00:00:00"
    mem=5
    lab=true
    """,  # Note: Manually set "pseudonym" to task_name in sbatch.yaml
]


runs.UnifiedML.plots = [
    ['DCGAN']
]

runs.UnifiedML.sftp = True
runs.UnifiedML.bluehive = False
runs.UnifiedML.lab = True
runs.UnifiedML.title = 'DCGAN'
