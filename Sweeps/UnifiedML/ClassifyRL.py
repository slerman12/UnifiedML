runs = {'UnifiedML': {
    'sweep': [
        # Classify + RL  EMA breaks discrete RL :o ! TODO
        """python Run.py
        Agent=Agents.ExperimentAgent 
        task=classify/mnist,classify/cifar10,classify/tinyimagenet
        ema=true 
        weight_decay=0.01
        Eyes=Blocks.Architectures.ResNet18
        'Aug="RandomShiftsAug(4)"'
        'transform="{RandomHorizontalFlip:{}}"'
        RL=true 
        supervise=false
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
        ['Classify+RL_supervise-False.*', 'Classify'],
    ],
    'sftp': False,
    'bluehive': False,
    'lab': True,
    'steps': 2000000,
    'title': 'UnifiedML',
    'x_axis': 'Step',
    'bluehive_only': [],
    'tasks': [],
    'agents': [],
    'suites': [],
    'write_tabular': False}
}