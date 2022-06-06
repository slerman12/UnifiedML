# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import getpass
import os
from pathlib import Path
from cryptography.fernet import Fernet

from pexpect import spawn

import numpy as np

from Plot import plot


sftp = True
plot_group = 'UML_Paper'
# steps = None
steps = 5e5

plots = [
    # Generalized reference implementations: DQN, DrQV2, SPR
    ['Self-Supervised', 'DQN-Based', 'Reference', 'Critic-Ensemble'],

    # Quickly integrating and prototyping computer vision advances in RL:
    # EMA, weight decay, augmentations, & architectures (e.g. ViT)
    ['CV-RL', 'ViT', 'Reference', 'CV-Transform-RL'],

    # Can RL augment supervision?
    ["No-Contrastive", "Half-Half-Contrastive", "Third-Label",
     'Actor-Experts', 'Supervised'],

    # When is reward enough?
    ["No-Contrastive-Pure-RL", "Half-Half-Contrastive-Pure-RL", "Third-Label-Pure-RL",
     'Supervised'],

    # Unifying RL as a discrete control problem: AC2
    ['Actor-Ensemble-3', 'Actor-Ensemble-5', 'DQN-Based',
     'Actions-Sampling-3', 'Actions-Sampling-5', 'Actor-Critic-Ensemble-5-5',
     'Reference']
]

bluehive_only = ["Half-Half-Contrastive", "Third-Label",
                 "Half-Half-Contrastive-Pure-RL", "Third-Label-Pure-RL",
                 'Actions-Sampling-3', 'Actions-Sampling-5', 'Actor-Critic-Ensemble-5-5',
                 'Critic-Ensemble', 'Reference']

agents = []

suites = []

tasks = ['cheetah_run', 'quadruped_walk', 'reacher_easy', 'cup_catch', 'finger_spin', 'walker_walk',
         'pong', 'breakout', 'boxing',
         # 'krull', 'seaquest', 'qbert',
         'cifar10', 'tinyimagenet']

plot_group = 'XRD'
plots = [['PS1_to_RRUFF', 'PS1_to_RRUFF_ResNet18', 'PS1_noise_20_to_RRUFF_ResNet18']]
tasks =[]

experiments = set().union(*plots)


# SFTP experiment results
if sftp:
    username = 'slerman'

    # Get password, encrypt, and save for reuse
    if os.path.exists('pass'):
        with open('pass', 'r') as file:
            key, encoded = file.readlines()
            password = Fernet(key).decrypt(bytes(encoded, 'utf-8'))
    else:
        password, key = getpass.getpass(), Fernet.generate_key()
        encoded = Fernet(key).encrypt(bytes(password, 'utf-8'))
        with open('pass', 'w') as file:
            file.writelines([key.decode('utf-8') + '\n', encoded.decode('utf-8')])

    conda = 'source /scratch/slerman/miniconda/bin/activate agi'

    # Connect VPN
    try:
        p = spawn('/opt/cisco/anyconnect/bin/vpn connect vpnconnect.rochester.edu')
        p.expect('Username: ')
        p.sendline('')
        p.expect('Password: ')
        p.sendline(password)
        p.expect('Second Password: ')
        p.sendline('push')
        p.expect('VPN>')
    except Exception:
        pass

    # SFTP

    print(f'SFTP\'ing: {", ".join(experiments)}')
    if len(tasks):
        print(f'plotting for tasks: {", ".join(tasks)}')
    if steps:
        print(f'up to steps: {steps:.0f}')

    cwd = os.getcwd()
    local_path = f"./Benchmarking"

    Path(local_path).mkdir(parents=True, exist_ok=True)
    os.chdir(local_path)

    print('\nConnecting to Bluehive', end=" ")
    p = spawn(f'sftp {username}@bluehive.circ.rochester.edu')
    p.expect('Password: ', timeout=None)
    p.sendline(password)
    p.expect('sftp> ', timeout=None)
    print('- Connected! ✓\n')
    p.sendline(f"lcd {local_path}")
    p.expect('sftp> ', timeout=None)
    p.sendline(f"cd /scratch/{username}/UnifiedML")
    p.expect('sftp> ', timeout=None)
    for i, experiment in enumerate(experiments):
        print(f'{i + 1}/{len(experiments)} [bluehive] SFTP\'ing "{experiment}"')
        p.sendline(f"get -r ./Benchmarking/{experiment}")
        p.expect('sftp> ', timeout=None)

    print('\nConnecting to lab', end=" ")
    p = spawn(f'sftp cornea')
    p.expect('sftp> ')
    print('- Connected! ✓\n')
    p.sendline(f"lcd {local_path}")
    p.expect('sftp> ')
    p.sendline("cd UnifiedML")
    p.expect('sftp> ')
    for i, experiment in enumerate(experiments):
        if experiment not in bluehive_only:
            print(f'{i + 1}/{len(experiments)} [lab] SFTP\'ing "{experiment}"')
            p.sendline(f"get -r ./Benchmarking/{experiment}")
            p.expect('sftp> ')

    print('\nPlotting results...')

    os.chdir(cwd)

# Generate each plot
for plot_experiments in plots:

    plot(path=f"./Benchmarking/{plot_group + '/' if plot_group else ''}{'_'.join(plot_experiments)}/Plots",
         plot_experiments=plot_experiments if len(plot_experiments) else None,
         plot_agents=agents if len(agents) else None,
         plot_suites=suites if len(suites) else None,
         plot_tasks=tasks if len(tasks) else None,
         steps=steps if steps else np.inf, write_tabular=True, verbose=True)
