from Sweeps.Templates import template, atari_26


runs = template('UnifiedML')

runs.UnifiedML.sweep = [
    # Longer Exploration, Shorter "nstep"
    f"""
    task={atari_26}
    train_steps=3000000 
    save_per_steps=500000 
    replay.save=true
    'stddev_schedule="linear(1.0,0.1,800000)"'
    frame_stack=4
    nstep=3
    Agent=Agents.AC2Agent 
    experiment=Atari26
    time="5-00:00:00"
    lab=true
    mem=50
    """,
]  # Replay capacity is  1000000


runs.UnifiedML.plots = [
    ['Atari26'],
]

runs.UnifiedML.sftp = True
runs.UnifiedML.title = 'Atari26'

