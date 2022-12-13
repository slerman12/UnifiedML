from Sweeps.Templates import template, atari_tasks as atari


def join(atari_tasks):
    return f'atari/{",atari/".join([a.lower() for a in atari_tasks])}'


runs = template('UnifiedML')

runs.UnifiedML.sweep = [
    # Longer Exploration
    f"""
    task={join(atari[5:])}
    train_steps=1000000 
    save_per_steps=500000 
    replay.save=true
    'stddev_schedule="linear(1.0,0.1,800000)"'
    Agent=Agents.AC2Agent 
    experiment=Atari26
    time="5-00:00:00"
    reservation_id=20221217
    """,
]  # Replay capacity is  1000000


runs.UnifiedML.plots = [
    ['Atari26'],
]

runs.UnifiedML.sftp = True
runs.UnifiedML.lab = False
runs.UnifiedML.title = 'Atari26'
