from Sweeps.Templates import atari_26


atari_30 = atari_26 + ',atari/donkeykong,atari/spaceinvaders,atari/tictactoe3d,atari/videochess'

runs = {'core': {
    'sweep': [
        f'train_steps=2000000 '
        f'task={atari_30} '
        f'experiment="Atari-30" '
        f'logger.wandb=true '
        f'reservation_id=20220929 ',

        f'train_steps=2000000 '
        f'task=mario '
        f'experiment="Mario" '
        f'logger.wandb=true '
        f'reservation_id=20220929 '
    ],
    'plots': [
        ['Atari-30'],
    ],
    'sftp': True,
    'bluehive': True,
    'lab': False,
    'steps': 5e5,
    'title': 'UnifiedML',
    'x_axis': 'Step',
    'bluehive_only': [],
    'tasks': [],
    'agents': [],
    'suites': []}
}
