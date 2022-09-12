runs = {'Discrete-As-Continuous': {
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
    'lab': True,
    'steps': 5e5,
    'title': 'UML_Paper',
    'x_axis': 'Step',
    'bluehive_only': [],
    'tasks': [],
    'agents': [],
    'suites': []},
}
