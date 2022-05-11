# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import getpass
import os
from cryptography.fernet import Fernet

from pexpect import pxssh, spawn


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

conda = f'source /scratch/{username}/miniconda/bin/activate agi'  # TODO different CUDA per GPU

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

# Define sweep
dmc = 'dmc/cheetah_run,dmc/quadruped_walk,dmc/reacher_easy,dmc/cup_catch,dmc/finger_spin,dmc/walker_walk'
atari = 'atari/pong,atari/breakout,atari/boxing,atari/krull,atari/seaquest,atari/qbert'
sweep = [
    # f'Agent=Agents.DQNAgent,Agents.SPRAgent train_steps=100000 seed=1,2,3 task={atari} experiment="DQN-Based" '
    # f'plot_per_steps=0 reservation_id=20220502',
    # f'Agent=Agents.SPRAgent train_steps=500000 seed=1,2,3 task={dmc} experiment="Self-Supervised" plot_per_steps=0 reservation_id=20220502',
    # f'Agent=Agents.DQNAgent +agents.num_critics=5 train_steps=100000 seed=1,2,3 task={atari},{dmc} experiment="Critic-Ensemble" plot_per_steps=0 reservation_id=20220502',
    # f'Agent=Agents.AC2Agent +agents.num_actors=3,5 train_steps=100000 seed=1,2,3 task={atari} experiment="Actor-Ensemble" plot_per_steps=0 reservation_id=20220502',
    # f'Agent=Agents.AC2Agent +agents.num_actions=3,5 train_steps=100000 seed=1,2,3 task={atari} experiment="Actions-Sampling" plot_per_steps=0'
    # f'Agent=Agents.AC2Agent +agents.num_actions=3,5 train_steps=500000 seed=1,2,3 task={dmc} experiment="Actions-Sampling" plot_per_steps=0 reservation_id=20220502',
    # f'Agent=Agents.DrQV2Agent train_steps=500000 seed=1,2,3 task={dmc} ema=true weight_decay=0.01 experiment="CV-RL" plot_per_steps=0 reservation_id=20220502',
    # f'Agent=Agents.DrQV2Agent train_steps=500000 seed=1,2,3 task={dmc} '
    # 'transform="{RandomCrop:{pad:4}}" Aug=Blocks.Architectures.Null experiment="CV-Transform-RL" plot_per_steps=0 reservation_id=20220502',
    # f'Agent=Agents.DrQV2Agent train_steps=500000 seed=1,2,3 task={dmc} Eyes=Blocks.Architectures.ViT experiment="ViT" plot_per_steps=0 reservation_id=20220502',
    # 'task=classify/cifar10,classify/tinyimagenet RL=false ema=true weight_decay=0.01 transform="{RandomHorizontalFlip:{}}" experiment="Supervised" plot_per_steps=0 num_workers=16 num_gpus=4 parallel=true reservation_id=20220502',
    # 'task=classify/cifar10,classify/tinyimagenet ema=true weight_decay=0.01 transform="{RandomHorizontalFlip:{}}" experiment="Supervised+RL" plot_per_steps=0 num_workers=16 num_gpus=4 parallel=true reservation_id=20220502',
    'Agent=Agents.AC2Agent +agents.num_actors=5 task=classify/tinyimagenet ema=true weight_decay=0.01 transform="{RandomHorizontalFlip:{}}" experiment="Actor-Experts" plot_per_steps=0 num_workers=16 num_gpus=4 parallel=true reservation_id=20220502'
]
atari_tasks = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
    'Seaquest', 'UpNDown'
]
sweep = ['experiment=Random Agent=Agents.RandomAgent train_steps=0 evaluate_episodes=100 '
         f'task=atari/{",atari/".join([a.lower() for a in atari_tasks])} '
         'num_workers=4 num_gpus=1 mem=20 gpu="K80" '
         'plot_per_steps=0 reservation_id=20220509']


# Launch on Bluehive
try:
    s = pxssh.pxssh()
    s.login('bluehive.circ.rochester.edu', username, password)
    s.sendline(f'cd /scratch/{username}/UnifiedML')     # Run a command
    s.prompt()                                          # Match the prompt
    print(s.before)                                     # Print everything before the prompt.
    s.sendline('git pull origin master')
    s.prompt()
    print(s.before)
    s.sendline(conda)
    s.prompt()
    print(s.before)
    for hyperparams in sweep:
        print(f'python sbatch.py -m {hyperparams} username="{username}" conda="{conda}"')
        s.sendline(f'python sbatch.py -m {hyperparams} username="{username}" conda="{conda}"')
        s.prompt()
        print(s.before)
    s.logout()
except pxssh.ExceptionPxssh as e:
    print("pxssh failed on login.")
    print(e)
