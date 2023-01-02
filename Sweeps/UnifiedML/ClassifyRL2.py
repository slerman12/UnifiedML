# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Sweeps.Templates import template


runs = template('UnifiedML')

runs.UnifiedML.sweep = [
    # If synchronization worked.. Maybe try this (semaphores):
    # https://stackoverflow.com/questions/16654908/synchronization-across-multiple-processes-in-python

    # All experiments
    f"""
    task=classify/mnist,classify/cifar10,classify/tinyimagenet
    train_steps=500000 
    save_per_steps=100000 
    replay.save=true
    online=true
    stream=false,true
    weight_decay=0.01
    Eyes=Blocks.Architectures.ResNet18
    'Aug="RandomShiftsAug(4)"'
    'transform="{{RandomHorizontalFlip:{{}}}}"'
    RL=true 
    supervise=false
    discrete=false,true 
    experiment='ClassifyRL_online-${{online}}_stream-${{stream}}_discrete-${{discrete}}'
    plot_per_steps=0
    time="5-00:00:00"
    mem=50
    lab=true
    """,  # Note: Manually set "pseudonym" to task_name in sbatch.yaml
]

# runs.UnifiedML.sweep = [
#     # Had To Create Replays ...  On Bluehive, didn't sync.
#     f"""
#     task=classify/mnist,classify/cifar10,classify/tinyimagenet
#     train_steps=500000
#     save_per_steps=100000
#     replay.save=true
#     online=true
#     stream=false
#     weight_decay=0.01
#     Eyes=Blocks.Architectures.ResNet18
#     'Aug="RandomShiftsAug(4)"'
#     'transform="{{RandomHorizontalFlip:{{}}}}"'
#     RL=true
#     supervise=false
#     discrete=false
#     experiment='ClassifyRL_online-${{online}}_stream-${{stream}}_discrete-${{discrete}}'
#     plot_per_steps=0
#     time="5-00:00:00"
#     mem=50
#     reservation_id=20221217
#     """,
#
#     # # Remainder Of Experiments
#     # f"""
#     # task=classify/mnist,classify/cifar10,classify/tinyimagenet
#     # train_steps=500000
#     # save_per_steps=100000
#     # replay.save=true
#     # online=true
#     # stream=false
#     # weight_decay=0.01
#     # Eyes=Blocks.Architectures.ResNet18
#     # 'Aug="RandomShiftsAug(4)"'
#     # 'transform="{{RandomHorizontalFlip:{{}}}}"'
#     # RL=true
#     # supervise=false
#     # discrete=True
#     # experiment='ClassifyRL_online-${{online}}_stream-${{stream}}_discrete-${{discrete}}'
#     # plot_per_steps=0
#     # time="5-00:00:00"
#     # mem=50
#     # reservation_id=20221217
#     # """,
#     #
#     # # Remainder Of Experiments
#     # f"""
#     # task=classify/mnist,classify/cifar10,classify/tinyimagenet
#     # train_steps=500000
#     # save_per_steps=100000
#     # replay.save=true
#     # online=true
#     # stream=true
#     # weight_decay=0.01
#     # Eyes=Blocks.Architectures.ResNet18
#     # 'Aug="RandomShiftsAug(4)"'
#     # 'transform="{{RandomHorizontalFlip:{{}}}}"'
#     # RL=true
#     # supervise=false
#     # discrete=false,true
#     # experiment='ClassifyRL_online-${{online}}_stream-${{stream}}_discrete-${{discrete}}'
#     # plot_per_steps=0
#     # time="5-00:00:00"
#     # mem=50
#     # reservation_id=20221217
#     # """,  # Note: Manually set "pseudonym" to task_name in sbatch.yaml
# ]  # Replay capacity is  1000000, Note: Crucial: Had to set experiment='...' in quotes for interpolation to work


runs.UnifiedML.plots = [
    ['ClassifyRL_online.*'],
]

runs.UnifiedML.sftp = True
runs.UnifiedML.bluehive = False  # Also running on Bluehive
runs.UnifiedML.lab = True  # Also running on lab

runs.UnifiedML.title = 'Classify Via Reinforcement Learning'
runs.UnifiedML.steps = 5e5
