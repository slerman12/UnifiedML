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
    # Next time: Load via replay.load=true.
    'sweep': [
        f'train_steps=2500000 '  # Changed from 2000000 to add 500000
        f'task={atari_26} '
        f'experiment="Atari-26-DQN" '  # Originally did 30 - Meant to write SoftDQN
        f'logger.wandb=true '
        f'time="12-00:00:00" '
        f'save_per_steps=500000 '
        f'replay.save=true '
        f'reservation_id=20220929 '
        f'load=false '
        f'replay.load=false ',

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
        ['Atari-26-DQN'],  # I named it inconsistently
        ['Mario'],
        ['Self-Supervised_Mario'],
        ['Atari-26_Continuous'],
        ['Atari-26_Continuous', 'Atari-30'],
        ['Mario', 'Self-Supervised_Mario'],
    ],
    'sftp': True,
    'bluehive': True,
    'lab': False,
    'write_tabular': True,
    'steps': 2500000,
    'title': 'Mario',
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
