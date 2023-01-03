# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Sweeps.Templates import template, atari, dmc


def join_dmc(dmc_tasks):
    return f'dmc/{",dmc/".join([t.lower() for t in dmc_tasks])}'


def join_atari(atari_tasks):
    return f'atari/{",atari/".join([a.lower() for a in atari_tasks])}'


runs = template('UnifiedML')

runs.UnifiedML.sweep = [
    # Less Exploration
    f"""
    task={join_atari(atari)},{join_dmc(dmc)}
    train_steps=1000000
    save_per_steps=200000
    replay.save=true
    Agent=Agents.DrQV2Agent,Agents.DQNAgent
    experiment=ActionSpaces
    time="5-00:00:00"
    mem=50
    reservation_id=20221217
    """,  # Note: Manually set "pseudonym" to task_name in sbatch.yaml
]  # Replay capacity is  1000000


runs.UnifiedML.plots = [
    ['ActionSpaces']
]

# runs.UnifiedML.agents = ['DrQV2Agent']

runs.UnifiedML.sftp = True
runs.UnifiedML.lab = False
runs.UnifiedML.title = 'Action Space Adaptation, Discrete <-> Continuous'
runs.UnifiedML.steps = 1e6
runs.UnifiedML.write_tabular = True
