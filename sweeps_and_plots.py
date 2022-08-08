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
        -> Sweep & Sweep Metadata
    ...
    -> Sweep Group M:
        -> Sweep & Sweep Metadata
...
-> Project N:
    -> ...
    
Sweep groups allow more fine-grained organization of project-specific runs.
"""

runs = {
    'Template':
        {
            'Sweep1': {
                'sweep': [
                    # Sweep commands go here
                ],
                'plots': [
                    [],
                ],
                'sftp': True,
                'bluehive': True,
                'steps': 5e5,
                'title': 'Template',
                'x_axis': 'Step',
                'bluehive_only': [],
                'tasks': [],
                'agents': [],
                'suites': []},
        },
    'UML_Paper':
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
                    lr_decay_epochs=100
                    Eyes=Blocks.Architectures.ResNet18
                    'transform="{RandomHorizontalFlip:{}}"'
                    RL=true 
                    supervise=false,true 
                    discrete=true,false 
                    +agent.half=true,false 
                    experiment='Classify+RL_supervise-${supervise}_discrete-${discrete}_contrastive-${agent.half}' 
                    logger.wandb=true
                    parallel=true
                    num_workers=20
                    plot_per_steps=0""",

                    # Classify + RL: Variational Inference
                    """python Run.py
                    Agent=Agents.ExperimentAgent 
                    task=classify/mnist,classify/cifar10,classify/tinyimagenet
                    ema=true 
                    weight_decay=0.01
                    lr_decay_epochs=100
                    Eyes=Blocks.Architectures.ResNet18
                    RL=true 
                    supervise=false,true 
                    discrete=false 
                    +agent.half=true,false 
                    +agent.sample=true
                    experiment='Classify+RL+Sample_supervise-${supervise}_discrete-${discrete}_contrastive-${agent.half}' 
                    logger.wandb=true
                    parallel=true
                    num_workers=20
                    plot_per_steps=0"""
                ],
                'plots': [
                    # Classify + RL
                    ['Classify+RL.*'],

                    # Q-learning expected, expected + entropy, best -- Note: No sweep for this one since modified Losses
                    ['Q-Learning-Target.*'],
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
        },
    'XRD':
        {
            'Summary': {
                'sweep': [
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
                    # Summary bar & line plots
                    ['CNN_optim_.*', 'MLP_optim_.*'],
                ],
                'sftp': True,
                'bluehive': False,
                'steps': 5e5,
                'title': 'RRUFF',
                'x_axis': 'Step',
                'bluehive_only': [],
                'tasks': [],
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


runs = convert_to_attr_dict(runs)