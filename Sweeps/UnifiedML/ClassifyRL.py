runs = {'UnifiedML': {
    'sweep': [
        # Classify + RL  EMA breaks discrete RL :o ! TODO
        # """python Run.py
        # Agent=Agents.ExperimentAgent
        # task=classify/mnist,classify/cifar10,classify/tinyimagenet
        # ema=true
        # weight_decay=0.01
        # Eyes=Blocks.Architectures.ResNet18
        # 'Aug="RandomShiftsAug(4)"'
        # 'transform="{RandomHorizontalFlip:{}}"'
        # RL=true
        # supervise=false
        # discrete=true,false
        # +agent.contrastive=true,false
        # experiment='Classify+RL_supervise-${supervise}_discrete-${discrete}_contrastive-${agent.contrastive}'
        # num_workers=8
        # save_per_steps=100000
        # plot_per_steps=0""",

        # Classify + RL  EMA breaks discrete RL - no EMA
        # """Agent=Agents.ExperimentAgent
        # task=classify/mnist,classify/cifar10,classify/tinyimagenet
        # ema=false
        # weight_decay=0.01
        # Eyes=Blocks.Architectures.ResNet18
        # 'Aug="RandomShiftsAug(4)"'
        # 'transform="{RandomHorizontalFlip:{}}"'
        # RL=true
        # supervise=false
        # discrete=true,false
        # +agent.contrastive=true,false
        # experiment='Classify+RL_supervise-${supervise}_discrete-${discrete}_contrastive-${agent.contrastive}_no-EMA'
        # num_workers=8
        # plot_per_steps=0
        # save_per_steps=100000""",

        # Classify + RL: Variational Inference - No EMA
        # """Agent=Agents.ExperimentAgent
        # task=classify/mnist,classify/cifar10,classify/tinyimagenet
        # ema=false
        # weight_decay=0.01
        # Eyes=Blocks.Architectures.ResNet18
        # 'Aug="RandomShiftsAug(4)"'
        # 'transform="{RandomHorizontalFlip:{}}"'
        # RL=true
        # supervise=false
        # discrete=true,false
        # +agent.contrastive=true,false
        # +agent.sample=true
        # +agent.num_critics=1,2
        # experiment='Classify+RL+Sample_supervise-${supervise}_discrete-${discrete}_contrastive-${agent.contrastive}_num-critics-${agent.num_critics}_no-EMA'
        # num_workers=8
        # plot_per_steps=0
        # logger.wandb=true
        # time="12-00:00:00"
        # reservation_id=20220929
        # save_per_steps=100000""",

        # Classify + RL: Variational Inference - No EMA - need to redo discrete
        """Agent=Agents.ExperimentAgent 
        task=classify/tinyimagenet
        ema=false 
        weight_decay=0.01
        Eyes=Blocks.Architectures.ResNet18
        'Aug="RandomShiftsAug(4)"'
        'transform="{RandomHorizontalFlip:{}}"'
        RL=true 
        supervise=false
        discrete=False 
        +agent.contrastive=true,false 
        +agent.sample=true
        +agent.num_critics=2
        experiment='Classify+RL+Sample_supervise-${supervise}_discrete-${discrete}_contrastive-${agent.contrastive}_num-critics-${agent.num_critics}_no-EMA'
        num_workers=8
        plot_per_steps=0
        logger.wandb=true
        time="12-00:00:00" 
        reservation_id=20220929
        load=true
        save_per_steps=100000""",

        # Classify
        # """python Run.py
        # Agent=Agents.ExperimentAgent
        # task=classify/mnist,classify/cifar10,classify/tinyimagenet
        # ema=true
        # weight_decay=0.01
        # Eyes=Blocks.Architectures.ResNet18
        # 'Aug="RandomShiftsAug(4)"'
        # 'transform="{RandomHorizontalFlip:{}}"'
        # experiment='Classify'
        # num_workers=4
        # plot_per_steps=0
        # save_per_steps=100000"""
    ],
    'plots': [
        # Classify + RL
        ['Classify+RL_supervise-False.*true', 'Classify+RL_supervise-False.*false', 'Classify'],
        ['Classify+RL+Sample_supervise-False.*no-EMA', 'Classify_no-EMA'],
        ['Classify+RL_supervise-False.*no-EMA', 'Classify_no-EMA'],
    ],
    'sftp': True,
    'bluehive': True,
    'lab': True,
    'steps': 200000,
    'title': 'UnifiedML',
    'x_axis': 'Step',
    'bluehive_only': [],
    'tasks': ['mnist', 'cifar10'],  # TODO Note: adding classify/ broke te regex matching
    'agents': [],
    'suites': [],
    'write_tabular': False}
}