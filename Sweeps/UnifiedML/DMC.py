# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from Sweeps.Templates import template, dmc


def join(dmc_tasks):
    return f'dmc/{",dmc/".join([t.lower() for t in dmc_tasks])}'


runs = template('UnifiedML')

runs.UnifiedML.sweep = [
    # Longer Exploration
    f"""
    task={join(dmc)}
    train_steps=1000000 
    save_per_steps=200000 
    replay.save=true
    Agent=Agents.AC2Agent 
    experiment=DMC-6
    time="5-00:00:00"
    mem=50
    reservation_id=20221217
    """,  # Note: Manually set "pseudonym" to task_name in sbatch.yaml
]  # Replay capacity is  1000000


runs.UnifiedML.plots = [
    ['DMC-6'],
]

runs.UnifiedML.sftp = True
runs.UnifiedML.lab = False
runs.UnifiedML.title = 'DMC-6 From Images'
runs.UnifiedML.steps = 1e6
# runs.UnifiedML.write_tabular = True
