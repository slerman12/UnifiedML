from Sweeps.Templates import atari_26


atari_28 = atari_26 + ',atari/tictactoe3d,atari/videochess'

atari_tasks = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
    'Seaquest', 'UpNDown'
]
atari_26 = f'atari/{",atari/".join([a.lower() for a in atari_tasks])}'

atari_retry = [
    'ChopperCommand', 'Gopher'
]
atari_retry = f'atari/{",atari/".join([a.lower() for a in atari_retry])}'

runs = {'UnifiedML': {

    # Corrupted Checkpoint + after 2000000 Steps. Saved Agent but not Replay. Loaded Agent at 2000000 with empty Replay.
    # Need To: Save Replay via replay.save=true.
    # Next time: Load via replay.load=true.  -  just re-ran by mistake so latest replay is incorrect - replay corrupted!
    'sweep': [
        f'train_steps=3000000 '  # Changed from 2000000 to add 500000  TODO NOTE discrete= naming borke; it overrided
        f'task={atari_26} '
        f'experiment="Atari-26-SoftDQN-discrete-false" '  # Originally did 30 - Meant to write -SoftDQN - wrote -DQN
        f'logger.wandb=true '
        f'time="12-00:00:00" '
        f'save_per_steps=500000 '
        f'replay.save=true '
        f'reservation_id=20220929 '
        f'discrete=false '  # TODO I guess sweep breaks the naming reference
        f'Agent=Agents.AC2Agent '
        f'load=false '
        f'replay.load=false '
        f'\'stddev_schedule="linear(1.0,0.1,500000)"\' '  # Increased stddev
        f'replay.capacity=4000000 ',

        # f'train_steps=2000000 '
        # f'task=mario '
        # f'experiment="Mario" '
        # f'task_dir=mario '
        # f'logger.wandb=true '
        # f'time="12-00:00:00" '
        # f'save_per_steps=500000 '
        # f'reservation_id=20220929 '

        # f'train_steps=2000000 '
        # f'task={atari_26} '
        # f'experiment="Atari-26_Continuous" '  # Bash scripts seem to break on spaces; either use special char or sub _
        # f'Agent=Agents.AC2Agent '
        # f'discrete=false '
        # f'logger.wandb=true '
        # f'time="12-00:00:00" '
        # f'save_per_steps=500000 '
        # f'reservation_id=20220929 ',

        # f'train_steps=2000000 '
        # f'task=mario '
        # f'experiment="Self-Supervised_Mario" '  # Bash scripts seem to break on spaces; either use special char or sub _
        # f'Agent=Agents.AC2Agent '
        # f'+agent.depth=5 '
        # f'logger.wandb=true '
        # f'time="12-00:00:00" '
        # f'save_per_steps=500000 '
        # f'reservation_id=20220929 '
        # f'task_dir=mario '

        # Retry - some crashed I guess
        # f'train_steps=1500000 '
        # f'load=true '
        # f'task={atari_retry} '
        # f'experiment="Atari-30" '
        # f'logger.wandb=true '
        # f'time="12-00:00:00" '
        # f'save_per_steps=500000 '
        # f'reservation_id=20220929 ',
    ],
    'plots': [
        ['Atari-26-SoftDQN-discrete-true'],  # I named it inconsistently
        ['Atari-26-SoftDQN-discrete-false'],  # Naming broke
        # ['Mario'],
        # ['Self-Supervised_Mario'],
        # ['Atari-26_Continuous'],
        # ['Atari-26_Continuous', 'Atari-30'],
        # ['Mario', 'Self-Supervised_Mario'],
    ],
    'sftp': True,
    'bluehive': True,
    'lab': False,
    'write_tabular': True,
    'steps': 3000000,
    'title': 'Atari',
    'x_axis': 'Step',
    'bluehive_only': [],
    'tasks': ['Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
              'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
              'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
              'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
              'Seaquest', 'UpNDown', 'Mario'],
    # 'tasks': ['Mario'],
    'agents': [],
    'suites': []}
}
