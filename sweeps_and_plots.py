# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.

"""
Useful collections/stats/pre-defined sweeps
"""

dmc = 'dmc/cheetah_run,dmc/quadruped_walk,dmc/reacher_easy,dmc/cup_catch,dmc/finger_spin,dmc/walker_walk'

atari = 'atari/pong,atari/breakout,atari/boxing,atari/krull,atari/seaquest,atari/qbert'

atari_tasks = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
    'Seaquest', 'UpNDown'
]
full_atari = f'atari/{",atari/".join([a.lower() for a in atari_tasks])}'

"""
Structure of runs:

-> Project 1:
    -> Sweep Group 1:
        -> Sweep & Plots & Plots Metadata
    ...
    -> Sweep Group M:
        -> Sweep & Plots & Plots Metadata
...
-> Project N:
    -> ...
    
Sweep groups allow more fine-grained organization of project-respective runs.
"""


def template(name):
    return {
               name: {
                   'sweep': [
                       # Sweep commands go here
                   ],
                   'plots': [
                       # Sets of plots
                       [],
                   ],

                   # Plotting-related commands go here
                   'sftp': True,
                   'bluehive': True,
                   'steps': 5e5,
                   'title': 'Template',
                   'x_axis': 'Step',
                   'bluehive_only': [],
                   'tasks': [],
                   'agents': [],
                   'suites': []},
           }


runs = {
    'Template': template('template'),
    'UnifiedML':
        {
            'core': {
                'sweep': [
                    # f'Agent=Agents.DQNAgent,Agents.SPRAgent train_steps=100000 seed=1,2,3 task={atari} experiment="DQN-Based" '
                    # f'plot_per_steps=0 reservation_id=20220502',
                    # f'Agent=Agents.SPRAgent train_steps=500000 seed=1,2,3 task={dmc} experiment="Self-Supervised" plot_per_steps=0 reservation_id=20220502',
                    f'Agent=Agents.DQNAgent +agent.num_critics=5 train_steps=100000 seed=1,2,3 task={atari},{dmc} experiment="Critic-Ensemble" plot_per_steps=0 gpu="K80"',
                    # TODO
                    # f'Agent=Agents.AC2Agent +agent.num_actors=3 train_steps=500000 seed=1,2,3 task={dmc} experiment="Actor-Ensemble-3" plot_per_steps=0 gpu="K80"',
                    f'Agent=Agents.AC2Agent +agent.num_actors=5 +agent.num_critics=5 train_steps=500000 seed=1,2,3 task={dmc} experiment="Actor-Critic-Ensemble-5-5" plot_per_steps=0 gpu="K80"',
                    f'Agent=Agents.AC2Agent +agent.num_actions=5 +agent.num_actors=1 train_steps=500000 seed=1,2,3 task={dmc} experiment="Actions-Sampling-5" plot_per_steps=0 gpu="K80"',
                    f'Agent=Agents.AC2Agent +agent.num_actions=3 +agent.num_actors=1 train_steps=500000 seed=1,2,3 task={dmc} experiment="Actions-Sampling-3" plot_per_steps=0 gpu="K80"',
                    # f'Agent=Agents.DrQV2Agent train_steps=500000 seed=1,2,3 task={dmc} ema=true weight_decay=0.01 experiment="CV-RL" plot_per_steps=0 reservation_id=20220502',

                    # f'Agent=Agents.DrQV2Agent train_steps=500000 seed=1,2,3 task={dmc} '
                    # 'transform="{RandomCrop:{padding:4}}" recipes.Aug=Blocks.Architectures.Null experiment="CV-Transform-RL" plot_per_steps=0 gpu="K80"',
                    # f'Agent=Agents.DrQV2Agent train_steps=500000 seed=1,2,3 task={dmc} Eyes=Blocks.Architectures.ViT +recipes.encoder.eyes.patch_size=7 experiment="ViT" plot_per_steps=0 reservation_id=20220509',

                    # 'task=classify/cifar10,classify/tinyimagenet RL=false ema=true weight_decay=0.01 transform="{RandomHorizontalFlip:{}}" Eyes=Blocks.Architectures.ResNet18 experiment="Supervised" plot_per_steps=0 lab=true',

                    # 'task=classify/cifar10,classify/tinyimagenet ema=true weight_decay=0.01 transform="{RandomHorizontalFlip:{}}" Eyes=Blocks.Architectures.ResNet18 experiment="Supervised+RL" plot_per_steps=0 num_workers=16 num_gpus=4 parallel=true reservation_id=20220502',
                    # 'Agent=Agents.ExperimentAgent task=classify/cifar10,classify/tinyimagenet ema=true weight_decay=0.01 transform="{RandomHorizontalFlip:{}}" Eyes=Blocks.Architectures.ResNet18 experiment="Supervised-RL-No-Contrastive" plot_per_steps=0 num_workers=16 num_gpus=4 parallel=true gpu=K80|V100'
                    # 'Agent=Agents.AC2Agent +agent.num_actors=5 task=classify/cifar10 ema=true weight_decay=0.01 transform="{RandomHorizontalFlip:{}}" Eyes=Blocks.Architectures.ResNet18 experiment="Actor-Experts" RL=false plot_per_steps=0 num_workers=16 num_gpus=4 parallel=true'
                ],
                'plots': [
                    # Generalized reference implementations: DQN, DrQV2, SPR
                    ['Self-Supervised', 'DQN-Based', 'Reference', 'Critic-Ensemble'],

                    # Quickly integrating and prototyping computer vision advances in RL:
                    # EMA, weight decay, augmentations, & architectures (e.g. ViT)
                    ['CV-RL', 'ViT', 'Reference', 'CV-Transform-RL'],

                    # Can RL augment supervision?
                    ["No-Contrastive", "Half-Half-Contrastive", "Third-Label",
                     'Actor-Experts', 'Supervised'],

                    # When is reward enough?
                    ["No-Contrastive-Pure-RL", "Half-Half-Contrastive-Pure-RL", "Third-Label-Pure-RL",
                     'Supervised'],

                    # Unifying RL as a discrete control problem: AC2
                    ['Actor-Ensemble-3', 'Actor-Ensemble-5', 'DQN-Based',
                     'Actions-Sampling-3', 'Actions-Sampling-5', 'Actor-Critic-Ensemble-5-5',
                     'Reference'],
                ],
                'sftp': True,
                'bluehive': True,
                'steps': 5e5,
                'title': 'UML Paper',
                'x_axis': 'Step',
                'bluehive_only': ["Half-Half-Contrastive", "Third-Label",
                                  "Half-Half-Contrastive-Pure-RL", "Third-Label-Pure-RL",
                                  'Actions-Sampling-3', 'Actions-Sampling-5', 'Actor-Critic-Ensemble-5-5',
                                  'Critic-Ensemble', 'Reference'],
                'tasks': ['cheetah_run', 'quadruped_walk', 'reacher_easy', 'cup_catch', 'finger_spin', 'walker_walk',
                          'pong', 'breakout', 'boxing',
                          # 'krull', 'seaquest', 'qbert',
                          # 'mspacman', 'jamesbond', 'frostbite', 'demonattack', 'battlezone', 'alien', 'hero'
                          'cifar10', 'tinyimagenet'],
                'agents': [],
                'suites': []},
            'Classify+RL': {
                'sweep': [
                    # Classify + RL
                    """python Run.py
                    Agent=Agents.ExperimentAgent 
                    task=classify/mnist,classify/cifar10,classify/tinyimagenet
                    ema=true 
                    weight_decay=0.01
                    Eyes=Blocks.Architectures.ResNet18
                    'Aug="RandomShiftsAug(4)"'
                    'transform="{RandomHorizontalFlip:{}}"'
                    RL=true 
                    supervise=false,true 
                    discrete=true,false 
                    +agent.contrastive=true,false 
                    experiment='Classify+RL_supervise-${supervise}_discrete-${discrete}_contrastive-${agent.contrastive}' 
                    num_workers=8
                    plot_per_steps=0""",

                    # Classify + RL: Variational Inference
                    """python Run.py
                    Agent=Agents.ExperimentAgent 
                    task=classify/mnist,classify/cifar10,classify/tinyimagenet
                    ema=true 
                    weight_decay=0.01
                    Eyes=Blocks.Architectures.ResNet18
                    'Aug="RandomShiftsAug(4)"'
                    'transform="{RandomHorizontalFlip:{}}"'
                    RL=true 
                    supervise=false,true 
                    discrete=false 
                    +agent.contrastive=true,false 
                    +agent.sample=true
                    experiment='Classify+RL+Sample_supervise-${supervise}_discrete-${discrete}_contrastive-${agent.contrastive}'
                    num_workers=8
                    plot_per_steps=0""",

                    # Classify
                    """python Run.py
                    Agent=Agents.ExperimentAgent 
                    task=classify/mnist,classify/cifar10,classify/tinyimagenet
                    ema=true 
                    weight_decay=0.01
                    Eyes=Blocks.Architectures.ResNet18
                    'Aug="RandomShiftsAug(4)"'
                    'transform="{RandomHorizontalFlip:{}}"'
                    experiment='Classify' 
                    num_workers=4
                    plot_per_steps=0"""
                ],
                'plots': [
                    # Classify + RL
                    ['Classify+RL.*', 'Classify'],
                ],
                'sftp': True,
                'bluehive': False,
                'steps': 5e5,
                'title': 'UML Paper',
                'x_axis': 'Step',
                'bluehive_only': [],
                'tasks': [],
                'agents': [],
                'suites': []},

            'Discrete-As-Continuous': {
                'sweep': [
                    # Atari Un-Norm'd - Note: modified Actor
                    """python Run.py 
                    Agent=Agents.AC2Agent 
                    experiment=discrete-as-continuous-creator
                    discrete=false 
                    task=atari/pong,atari/breakout""",

                    # Atari Norm'd
                    """python Run.py 
                    Agent=Agents.AC2Agent 
                    experiment=discrete-as-continuous-creator-normed 
                    discrete=false 
                    task=atari/pong,atari/breakout""",

                    # Classify Un-Norm'd - Note: modified Actor
                    """python Run.py 
                    Agent=Agents.AC2Agent 
                    experiment=discrete-as-continuous-creator
                    discrete=false 
                    task=classify/mnist,classify/cifar10
                    ema=true 
                    weight_decay=0.01
                    Eyes=Blocks.Architectures.ResNet18
                    'Aug="RandomShiftsAug(4)"'
                    'transform="{RandomHorizontalFlip:{}}"'
                    RL=true 
                    supervise=false,true 
                    num_workers=4
                    plot_per_steps=0"""
                ],
                'plots': [
                    # Discrete as continuous + Creator, norm vs no-norm
                    ['discrete-as-continuous.*', 'Classify+RL_supervise-.*_discrete-False_contrastive-True'],

                    # Q-learning expected, expected + entropy, best -- Note: No sweep for this one, modified Losses
                    ['Q-Learning-Target.*'],
                ],
                'sftp': True,
                'bluehive': False,
                'steps': 5e5,
                'title': 'UML_Paper',
                'x_axis': 'Step',
                'bluehive_only': [],
                'tasks': [],
                'agents': [],
                'suites': []},

        },
    'XRD':
        {
            'Summary': {
                'sweep': [
                    # Large-Soup, 50-50, ViT, 7/230-Way, noise 0 - Launched ✓ (Cornea)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=ViT
                    Predictor=XRD.Predictor
                    batch_size=256
                    standardize=false
                    norm=true
                    task_name='Large-Soup-50-50_${dataset.num_classes}-Way'
                    experiment='ViT_optim_ADAM_batch_size_${batch_size}'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0.5]'
                    +dataset.num_classes=7,230
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Large-Soup, 50-50, No-Pool-CNN, 7/230-Way, noise 0 - Launched ✓ (Cornea)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=XRD.NoPoolCNN
                    Predictor=XRD.Predictor
                    batch_size=256
                    standardize=false
                    norm=true
                    task_name='Large-Soup-50-50_${dataset.num_classes}-Way'
                    experiment='No-Pool-CNN_optim_ADAM_batch_size_${batch_size}'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0.5]'
                    +dataset.num_classes=7,230
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Large-Soup, 50-50, CNN, 7/230-Way, noise 0 - Launched ✓ (Cornea)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=XRD.CNN
                    Predictor=XRD.Predictor
                    batch_size=256
                    standardize=false
                    norm=true
                    task_name='Large-Soup-50-50_${dataset.num_classes}-Way'
                    experiment='CNN_optim_ADAM_batch_size_${batch_size}'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0.5]'
                    +dataset.num_classes=7,230
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Large-Soup, 50-50, MLP, 7/230-Way, noise 0 - Launched ✓ (Cornea)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=Identity
                    Predictor=XRD.MLP
                    batch_size=256
                    standardize=false
                    norm=true
                    task_name='Large-Soup-50-50_${dataset.num_classes}-Way'
                    experiment='MLP_optim_ADAM_batch_size_${batch_size}'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd1.2m_large/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0.5]'
                    +dataset.num_classes=7,230
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Soup, 50-50, CNN, 7/230-Way, noise 0 - Launched ✓ (Cornea)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=XRD.CNN
                    Predictor=XRD.Predictor
                    batch_size=256,32
                    standardize=false
                    norm=false
                    task_name='Soup-50-50_${dataset.num_classes}-Way'
                    experiment='CNN_optim_ADAM_batch_size_${batch_size}'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0.5]'
                    +dataset.num_classes=7,230
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Soup, 50-50, CNN, 7-Way, noise 0, SGD, LR 1e-3, Batch Size 256 - Launched ✓ (Cornea)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=XRD.CNN
                    Predictor=XRD.Predictor
                    batch_size=256
                    Optim=SGD
                    lr=1e-3
                    standardize=false
                    norm=false
                    task_name='Soup-50-50_${dataset.num_classes}-Way'
                    experiment='CNN_optim_SGD_batch_size_${batch_size}_lr_1e-3'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0.5]'
                    +dataset.num_classes=7
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Soup, 50-50, MLP, 7/230-Way, noise 0 - Launched ✓ (Cornea)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=Identity
                    Predictor=XRD.MLP
                    batch_size=32,256
                    standardize=false
                    norm=false
                    task_name='Soup-50-50_${dataset.num_classes}-Way'
                    experiment='MLP_optim_ADAM_batch_size_${batch_size}'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0.5]'
                    +dataset.num_classes=7,230
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Soup, 50-50, MLP, 7-Way, SGD, LR 1e-3, Batch Size 256 - Launched ✓ (Cornea)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=Identity
                    Predictor=XRD.MLP
                    batch_size=256
                    standardize=false
                    norm=false
                    lr=1e-3
                    Optim=SGD
                    task_name='Soup-50-50_${dataset.num_classes}-Way'
                    experiment='MLP_optim_SGD_batch_size_${batch_size}_lr_1e-3'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0.5]'
                    +dataset.num_classes=7
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Soup, 50-50, CNN, 7/230-Way, noise 0, norm - Launched ✓ (Cornea)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=XRD.CNN
                    Predictor=XRD.Predictor
                    batch_size=256
                    standardize=false
                    norm=true
                    task_name='Soup-50-50_${dataset.num_classes}-Way'
                    experiment='CNN_optim_ADAM_batch_size_${batch_size}_norm'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0.5]'
                    +dataset.num_classes=7,230
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Soup, 50-50, CNN, 7/230-Way, noise 0, standardized - Launched ✓ (Cornea)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=XRD.CNN
                    Predictor=XRD.Predictor
                    batch_size=256
                    standardize=true
                    norm=false
                    task_name='Soup-50-50_${dataset.num_classes}-Way'
                    experiment='CNN_optim_ADAM_batch_size_${batch_size}_standardized'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0.5]'
                    +dataset.num_classes=7,230
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Soup, 50-50, CNN, 7-Way, noise 2 - Launched ✓ (Macula)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Blocks.Augmentations.IntensityAug
                    +aug.noise=2
                    Trunk=Identity
                    Eyes=XRD.CNN
                    Predictor=XRD.Predictor
                    batch_size=256
                    standardize=false
                    norm=false
                    task_name='Soup-50-50_${dataset.num_classes}-Way'
                    experiment='CNN_optim_ADAM_batch_size_${batch_size}_noise_2'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0.5]'
                    +dataset.num_classes=7
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Soup, 50-50, ResNet18, 7-Way, noise 0 - Launched X (Macula) - Too slow!
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=ResNet18
                    Predictor=XRD.Predictor
                    batch_size=256
                    standardize=false
                    norm=false
                    task_name='Soup-50-50_${dataset.num_classes}-Way'
                    experiment='ResNet18_optim_ADAM_batch_size_${batch_size}'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0.5]'
                    +dataset.num_classes=7
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Synthetic-Only, CNN, 7-Way, noise 0 - Launched ✓ (Cornea)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=XRD.CNN
                    Predictor=XRD.Predictor
                    batch_size=256
                    standardize=false
                    norm=false
                    task_name='Synthetic_${dataset.num_classes}-Way'
                    experiment='CNN_optim_ADAM_batch_size_${batch_size}'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0]'
                    +dataset.num_classes=7
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Synthetic-Only, MLP, 7-Way, noise 0 - Launched ✓ (Macula)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=Identity
                    Predictor=XRD.MLP
                    batch_size=256
                    standardize=false
                    norm=false
                    task_name='Synthetic_${dataset.num_classes}-Way'
                    experiment='MLP_optim_ADAM_batch_size_${batch_size}'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0]'
                    +dataset.num_classes=7
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Synthetic-Only, CNN, 7-Way, noise 0
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=XRD.CNN
                    Predictor=XRD.Predictor
                    batch_size=256
                    standardize=false
                    norm=false
                    Optim=SGD
                    lr=1e-3
                    task_name='Synthetic_${dataset.num_classes}-Way'
                    experiment='CNN_optim_SGD_batch_size_${batch_size}_lr_1e-3'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0]'
                    +dataset.num_classes=7
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",

                    # Synthetic-Only, MLP, 7-Way, noise 0 - Launched ✓ (Cornea)
                    """python Run.py
                    task=classify/custom
                    Dataset=XRD.XRD
                    Aug=Identity
                    Trunk=Identity
                    Eyes=Identity
                    Predictor=XRD.MLP
                    batch_size=256
                    standardize=false
                    norm=false
                    Optim=SGD
                    lr=1e-3
                    task_name='Synthetic_${dataset.num_classes}-Way'
                    experiment='MLP_optim_SGD_batch_size_${batch_size}_lr_1e-3'
                    '+dataset.roots=["../XRDs/icsd_Datasets/icsd171k_mix/","../XRDs/icsd_Datasets/rruff/XY_DIF_noiseAll/"]'
                    +'dataset.train_eval_splits=[1, 0]'
                    +dataset.num_classes=7
                    train_steps=5e5
                    save=true
                    logger.wandb=true""",
                ],
                'plots': [
                    # Summary bar & line plots  TODO rename Large-Pool -> "..._norm"
                    ['MLP_optim_ADAM_batch_size_256'],
                    # Summary bar & line plots  TODO rename Large-Pool -> "..._norm"
                    # ['.*CNN_optim_ADAM_batch_size_256',
                    #  '.*MLP_optim_ADAM_batch_size_256_norm',
                    #  'MLP_optim_ADAM_batch_size_256.*'],
                ],
                'sftp': True,
                'bluehive': False,
                'steps': 5e5,
                'title': 'RRUFF',
                'x_axis': 'Step',
                'bluehive_only': [],
                'tasks': ['.*7-Way.*'],
                'agents': [],
                'suites': []},
        }
}


# Access a dict with attribute or key (purely for aesthetic reasons)
class AttrDict(dict):
    def __init__(self, _dict):
        super(AttrDict, self).__init__()
        self.__dict__ = self
        self.update(_dict)


# Deep-converts an iterable into an AttrDict
def convert_to_attr_dict(iterable):
    if isinstance(iterable, dict):
        iterable = AttrDict(iterable)

    items = enumerate(iterable) if isinstance(iterable, (list, tuple)) \
        else iterable.items() if isinstance(iterable, AttrDict) else ()  # Iterate through lists, tuples, or dicts

    for key, value in items:
        iterable[key] = convert_to_attr_dict(value)  # Recurse through inner values

    return iterable


runs['XRD'].update(template('XRD_Final'))

runs = convert_to_attr_dict(runs)

runs.XRD.XRD_Final.plots = [
    ['MLP_optim_ADAM_batch_size_256.*'],
    ['.*CNN_optim_ADAM_batch_size_256.*'],
    ['ViT_optim_ADAM_batch_size_256'],
    ['ResNet18_optim_ADAM_batch_size_256'],
]
runs.XRD.XRD_Final.tasks = ['.*230-Way.*']
runs.XRD.XRD_Final.sftp = False
runs.XRD.XRD_Final.bluehive = False
runs.XRD.XRD_Final.title = 'RRUFF'

# TODO Could add .names to rename plot directories from experiment names to something more succinct


