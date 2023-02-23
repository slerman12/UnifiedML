# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Sweeps.Templates import template

runs = template('UnifiedML')

runs.UnifiedML.sweep = [
    # If synchronization worked.. Maybe try this (semaphores):
    # https://stackoverflow.com/questions/16654908/synchronization-across-multiple-processes-in-python

    # # Baseline
    # f"""
    # task=classify/mnist,classify/cifar10,classify/tinyimagenet
    # train_steps=500000
    # save_per_steps=100000
    # replay.save=true
    # weight_decay=0.01
    # Eyes=Blocks.Architectures.ResNet18
    # 'Aug="RandomShiftsAug(4)"'
    # 'transform="{{RandomHorizontalFlip:{{}}}}"'
    # ema=false,true
    # experiment='CrossEntropyBaseline_ema-${{ema}}'
    # plot_per_steps=0
    # time="5-00:00:00"
    # mem=50
    # lab=true
    # """,
    #
    # # Main experiments - no EMA
    # f"""
    # task=classify/mnist,classify/cifar10,classify/tinyimagenet
    # train_steps=500000
    # save_per_steps=100000
    # replay.save=true
    # weight_decay=0.01
    # Eyes=Blocks.Architectures.ResNet18
    # 'Aug="RandomShiftsAug(4)"'
    # 'transform="{{RandomHorizontalFlip:{{}}}}"'
    # RL=true
    # supervise=false
    # discrete=false,true
    # experiment='ClassifyRL_discrete-${{discrete}}'
    # plot_per_steps=0
    # time="5-00:00:00"
    # mem=50
    # lab=true
    # """,
    #
    # # EMA experiment
    # f"""
    # task=classify/mnist,classify/cifar10
    # train_steps=500000
    # save_per_steps=100000
    # replay.save=true
    # weight_decay=0.01
    # Eyes=Blocks.Architectures.ResNet18
    # 'Aug="RandomShiftsAug(4)"'
    # 'transform="{{RandomHorizontalFlip:{{}}}}"'
    # RL=true
    # supervise=false
    # discrete=false,true
    # ema=true,false
    # experiment='ClassifyRL_discrete-${{discrete}}_ema-${{ema}}'
    # plot_per_steps=0
    # time="5-00:00:00"
    # mem=50
    # lab=true
    # """,
    #
    # # Curse of dimensionality - obs shape
    # f"""
    # task=classify/tinyimagenet
    # train_steps=500000
    # save_per_steps=100000
    # replay.save=true
    # weight_decay=0.01
    # Eyes=Blocks.Architectures.ResNet18
    # 'Aug="RandomShiftsAug(4)"'
    # 'transform="{{RandomHorizontalFlip:{{}}}}"'
    # RL=true
    # supervise=false
    # discrete=false,true
    # 'env.transform="transforms.Compose([transforms.Resize(64),transforms.CenterCrop(64)])"'
    # experiment='ClassifyRL_discrete-${{discrete}}_size-28x28'
    # plot_per_steps=0
    # time="5-00:00:00"
    # mem=50
    # lab=true
    # """,

    # Curse of dimensionality - action shape  Note: This data overwrites the previous! TODO
    f"""
    task=classify/tinyimagenet
    train_steps=500000 
    save_per_steps=100000 
    replay.save=true
    weight_decay=0.01
    Eyes=Blocks.Architectures.ResNet18
    'Aug="RandomShiftsAug(4)"'
    'transform="{{RandomHorizontalFlip:{{}}}}"'
    RL=true 
    supervise=false
    discrete=false,true 
    'env.subset=[0,1,2,3,4,5,6,7,8,9]'
    experiment='ClassifyRL_discrete-${{discrete}}_classes-0-9'
    plot_per_steps=0
    time="5-00:00:00"
    mem=50
    lab=true
    """,  # Note: Manually set "pseudonym" to task_name in sbatch.yaml
]  # Note: Crucial: Had to set experiment='...' in quotes for interpolation to work


runs.UnifiedML.plots = [
    ['ClassifyRL_discrete.*', 'CrossEntropyBaseline.*'],
]

runs.UnifiedML.sftp = True
runs.UnifiedML.bluehive = True  # Also running on Bluehive
runs.UnifiedML.lab = False  # Also running on lab

runs.UnifiedML.title = 'Reinforcement Learner As Classifier'
runs.UnifiedML.steps = 5e5
