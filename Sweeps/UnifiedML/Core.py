from Sweeps.Templates import atari, dmc


runs = {'core': {
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
    'lab': True,
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
    'suites': []}
}
