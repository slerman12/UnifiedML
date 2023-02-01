# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Sweeps.Templates import template

runs = template('UnifiedML')

runs.UnifiedML.sweep = [
    # If synchronization worked.. Maybe try this (semaphores):
    # https://stackoverflow.com/questions/16654908/synchronization-across-multiple-processes-in-python

    # Replicas
    f"""
    task=classify/cifar10
    train_steps=100000 
    weight_decay=0.01
    Eyes=Blocks.Architectures.ResNet18
    'Aug="RandomShiftsAug(4)"'
    experiment='Replica_Per_Worker (stream)'
    stream=true
    plot_per_steps=0
    """,

    # Truly Shared RAM
    f"""
    task=classify/cifar10
    train_steps=100000 
    weight_decay=0.01
    Eyes=Blocks.Architectures.ResNet18
    'Aug="RandomShiftsAug(4)"'
    experiment=Truly_Shared_RAM
    plot_per_steps=0
    """,

    # Adaptive Memory Mapping
    f"""
    task=classify/cifar10
    train_steps=100000 
    weight_decay=0.01
    Eyes=Blocks.Architectures.ResNet18
    'Aug="RandomShiftsAug(4)"'
    experiment=Adaptive_Memory_Mapping_Hard_Disk
    capacity=30000
    plot_per_steps=0
    """,

    # Memory Mapping
    f"""
    task=classify/cifar10
    train_steps=100000 
    weight_decay=0.01
    Eyes=Blocks.Architectures.ResNet18
    'Aug="RandomShiftsAug(4)"'
    experiment=Adaptive_Memory_Mapping_Hard_Disk
    capacity=0
    plot_per_steps=0
    """,

    # Pytorch shared RAM (had to do manually)
    f"""
    task=classify/cifar10
    train_steps=100000 
    weight_decay=0.01
    Eyes=Blocks.Architectures.ResNet18
    'Aug="RandomShiftsAug(4)"'
    experiment=Pytorch_Shared_RAM
    plot_per_steps=0
    """,  # Note: Manually set "pseudonym" to task_name in sbatch.yaml
]


runs.UnifiedML.plots = [
    ['Replica.*', '.*Hard_Disk', '.*Shared_RAM'],
]

runs.UnifiedML.sftp = True
runs.UnifiedML.bluehive = False
runs.UnifiedML.lab = True  # Running on lab

runs.UnifiedML.title = 'Speed'
runs.UnifiedML.x_axis = 'Time'
