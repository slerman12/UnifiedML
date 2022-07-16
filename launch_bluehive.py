# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import getpass
import os
from cryptography.fernet import Fernet

from pexpect import pxssh, spawn


username = 'slerman'

branch = 'Epochs_And_Inits'

# Get password, encrypt, and save for reuse
if os.path.exists('pass'):
    with open('pass', 'r') as file:
        key, encoded = file.readlines()
        password = Fernet(key).decrypt(bytes(encoded, 'utf-8'))
else:
    password, key = getpass.getpass(), Fernet.generate_key()
    encoded = Fernet(key).encrypt(bytes(password, 'utf-8'))
    with open('pass', 'w') as file:
        file.writelines([key.decode('utf-8') + '\n', encoded.decode('utf-8')])

conda = f'source /scratch/{username}/miniconda/bin/activate agi'

# Connect VPN
try:
    p = spawn('/opt/cisco/anyconnect/bin/vpn connect vpnconnect.rochester.edu')
    p.expect('Username: ')
    p.sendline('')
    p.expect('Password: ')
    p.sendline(password)
    p.expect('Second Password: ')
    p.sendline('push')
    p.expect('VPN>')
except Exception:
    pass

# Define sweep  TODO evaluate_per_steps=2e4, move all this to separate file in dict with plot groups, wandb, no plot
dmc = 'dmc/cheetah_run,dmc/quadruped_walk,dmc/reacher_easy,dmc/cup_catch,dmc/finger_spin,dmc/walker_walk'
atari = 'atari/pong,atari/breakout,atari/boxing,atari/krull,atari/seaquest,atari/qbert'
sweep = [
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
]
sweep = [
    #      'task=classify/cifar10,classify/tinyimagenet ema=true weight_decay=0.01 '
    #          'Eyes=Blocks.Architectures.ResNet18 '
    #          'transform="{RandomHorizontalFlip:{}}" experiment="No-Contrastive-Pure-RL" '
    #          'Agent=Agents.ExperimentAgent '
    #          'parallel=true num_workers=20 num_gpus=4 mem=100 '
    #          'plot_per_steps=0 supervise=false lab=true',
    'task=classify/cifar10,classify/tinyimagenet ema=true weight_decay=0.01 '
    'Eyes=Blocks.Architectures.ResNet18 '
    'transform="{RandomHorizontalFlip:{}}" experiment="Half-Half-Contrastive-Pure-RL" '
    'Agent=Agents.ExperimentAgent +agent.half=true '
    'parallel=true num_workers=20 num_gpus=4 mem=100 '
    'plot_per_steps=0 supervise=false lab=true',
    'task=classify/cifar10,classify/tinyimagenet ema=true weight_decay=0.01 '
    'Eyes=Blocks.Architectures.ResNet18 '
    'transform="{RandomHorizontalFlip:{}}" experiment="Third-Label-Pure-RL" '
    'Agent=Agents.ExperimentAgent +agent.third=true '
    'parallel=true num_workers=20 num_gpus=4 mem=100 '
    'plot_per_steps=0 supervise=false lab=true'
    'task=classify/cifar10,classify/tinyimagenet ema=true weight_decay=0.01 '
    'Eyes=Blocks.Architectures.ResNet18 '
    'transform="{RandomHorizontalFlip:{}}" experiment="Half-Half-Contrastive" '
    'Agent=Agents.ExperimentAgent +agent.half=true '
    'parallel=true num_workers=20 num_gpus=4 mem=100 '
    'plot_per_steps=0 lab=true',
    'task=classify/cifar10,classify/tinyimagenet ema=true weight_decay=0.01 '
    'Eyes=Blocks.Architectures.ResNet18 '
    'transform="{RandomHorizontalFlip:{}}" experiment="Third-Label" '
    'Agent=Agents.ExperimentAgent +agent.third=true '
    'parallel=true num_workers=20 num_gpus=4 mem=100 '
    'plot_per_steps=0 lab=true'
]
atari_tasks = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
    'Seaquest', 'UpNDown'
]
full_atari = f'atari/{",atari/".join([a.lower() for a in atari_tasks])}'
# sweep = ['\'experiment="linear(2.0,0.1,40000)"\' \'stddev_schedule="linear(2.0,0.1,40000)"\' train_steps=100000 '
#          f'task=atari/pong,atari/breakout,atari/boxing,atari/krull,atari/seaquest,atari/qbert '
#          'num_workers=4 num_gpus=1 mem=20 '
#          'plot_per_steps=0 reservation_id=20220509']
sweep = ['"gpu=\'V100|A100\'" experiment=\'nvidia_smi_${gpu}\'']
sweep = ['task=dmc/cheetah_run gpu=K80,V100,A100 experiment=\'dmc_${gpu}_cudas\'',
         'task=dmc/cheetah_run gpu=\'RTX\' experiment=\'dmc_${gpu}_cudas\' lab=true']
sweep = ['task=dmc/cheetah_run gpu=A100,K80,V100 experiment=\'cuda_adaptive\'',
         'task=dmc/cheetah_run gpu=\'RTX\' experiment=\'cuda_adaptive\' lab=true']
sweep = [
    # 'task=classify/custom Dataset=\'Datasets.ReplayBuffer.Classify._RRUFF.RRUFF\' dataset.root=\'../XRDs/xrd_data/05_29_data/\' lab=true experiment=\'PS1_to_RRUFF\' Aug=Blocks.Architectures.Null num_gpus=4 parallel=true logger.wandb=true',
    # 'task=classify/custom Dataset=\'Datasets.ReplayBuffer.Classify._RRUFF.RRUFF\' +dataset.num_classes=230 dataset.name=\'230way\' dataset.root=\'../XRDs/xrd_data/05_29_data/\' experiment=\'PS1_to_RRUFF\' Aug=Blocks.Architectures.Null num_gpus=4 parallel=true logger.wandb=true',
    # 'task=classify/custom Dataset=\'Datasets.ReplayBuffer.Classify._RRUFF.RRUFF\' dataset.root=\'../XRDs/xrd_data/05_29_data/\' experiment=\'PS1_to_RRUFF_Random_Crop\' num_gpus=4 parallel=true logger.wandb=true',
    # 'task=classify/custom Dataset=\'Datasets.ReplayBuffer.Classify._RRUFF.RRUFF\' dataset.root=\'../XRDs/xrd_data/05_29_data/\' lab=true experiment=\'PS1_to_RRUFF_ResNet18\' Aug=Blocks.Architectures.Null Eyes=Blocks.Architectures.ResNet18 num_gpus=4 parallel=true logger.wandb=true',
    # 'task=classify/custom Dataset=\'Datasets.ReplayBuffer.Classify._RRUFF.RRUFF\' dataset.root=\'../XRDs/xrd_data/05_29_data/\' dataset.name=\'noise20\' experiment=\'PS1_noise20_to_RRUFF\' +dataset.data=\'icsd171k_ps1_noise20\' Aug=Blocks.Architectures.Null num_gpus=4 parallel=true logger.wandb=true',
    # 'task=classify/custom Dataset=\'Datasets.ReplayBuffer.Classify._RRUFF.RRUFF\' dataset.root=\'../XRDs/xrd_data/05_29_data/\' lab=true experiment=\'PS1_to_RRUFF_ViT\' Aug=Blocks.Architectures.Null Eyes=Blocks.Architectures.ViT +recipes.encoder.eyes.depth=6 +recipes.encoder.eyes.out_channels=128 num_gpus=8 parallel=true logger.wandb=true'
]
sweep = [
    # 'task=classify/custom Dataset=\'Datasets.ReplayBuffer.Classify._RRUFF.RRUFF\' dataset.root=\'../XRDs/xrd_data/\' dataset.name=\'mix\' +dataset.data=\'icsd171k_mix\' lab=true experiment=\'mix_to_RRUFF\' Aug=Blocks.Architectures.Null num_gpus=4 parallel=true logger.wandb=true',
    # 'task=classify/custom Dataset=\'Datasets.ReplayBuffer.Classify._RRUFF.RRUFF\' dataset.root=\'../XRDs/xrd_data/\' dataset.name=\'mix\' +dataset.data=\'icsd171k_mix\' lab=true experiment=\'mix_to_RRUFF_ResNet18\' Aug=Blocks.Architectures.Null Eyes=Blocks.Architectures.ResNet18 num_gpus=4 parallel=true logger.wandb=true',
    'task=classify/custom Dataset=\'Datasets.ReplayBuffer.Classify._RRUFF.RRUFF\' dataset.root=\'../XRDs/xrd_data/\' dataset.name=\'mix\' +dataset.data=\'icsd171k_mix\' lab=false experiment=\'mix_to_RRUFF_ResNet50\' Aug=Blocks.Architectures.Null Eyes=Blocks.Architectures.ResNet50 num_gpus=4 parallel=true logger.wandb=true',
]

sweep = [
    'python Run.py task=classify/custom Dataset=Datasets.ReplayBuffer.Classify._XRD.Synthetic "Pi_trunk=\'Null\'" Eyes=XRD.Encoder Pi_head=XRD.Actor Optim=torch.optim.SGD lr=0.001 batch_size=16 replay.forget=false replay.capacity=100 num_workers=1 \'aug="Null"\' logger.wandb=true experiment="Reproduced"'
]
# noinspection PyRedeclaration

# XRD

sweep = [
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

    # Soup, 50-50, ResNet18, 7-Way, noise 0 - Launched X (Macula) - Too big!
    # 1800-resolution input intractable, even on 8 parallelized A6000s apparently?
    # Or 2D->1D conversion attempts to allocate memory for 2d initially?
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
]


# Launch on Bluehive
try:
    s = pxssh.pxssh()
    s.login('bluehive.circ.rochester.edu', username, password)
    s.sendline(f'cd /scratch/{username}/UnifiedML')     # Run a command
    s.prompt()                                          # Match the prompt
    print(s.before.decode("utf-8"))                     # Print everything before the prompt.
    s.sendline(f'git fetch origin')
    s.prompt()
    print(s.before.decode("utf-8"))
    s.sendline(f'git checkout -b {branch} origin/{branch or "master"}')
    s.prompt()
    prompt = s.before.decode("utf-8")
    if f"fatal: A branch named '{branch}' already exists." in prompt:
        s.sendline(f'git checkout {branch}')
        s.prompt()
        prompt = s.before.decode("utf-8")
    print(prompt)
    assert 'error' not in prompt
    s.sendline(f'git pull origin {branch}')
    s.prompt()
    print(s.before.decode("utf-8"))
    s.sendline(conda)
    s.prompt()
    print(s.before.decode("utf-8"))
    for hyperparams in sweep:
        print(f'python sbatch.py -m {hyperparams} username="{username}"')
        s.sendline(f'python sbatch.py -m {hyperparams} username="{username}"')
        s.prompt()
        print(s.before.decode("utf-8"))
    s.logout()
except pxssh.ExceptionPxssh as e:
    print("pxssh failed on login.")
    print(e)
