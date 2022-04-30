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
sweep = ['task=classify/cifar10,classify/tinyimagenet ema=true weight_decay=0.01 '
         'Eyes=Blocks.Architectures.ResNet18 '
         'transform="{RandomHorizontalFlip:{}}" experiment="No-Contrastive" '
         'Agent=Agents.ExperimentAgent '
         'parallel=true num_workers=20 num_gpus=4 mem=100 '
         'plot_per_steps=0',
         'task=classify/cifar10,classify/tinyimagenet ema=true weight_decay=0.01 '
         'Eyes=Blocks.Architectures.ResNet18 '
         'transform="{RandomHorizontalFlip:{}}" experiment="Half-Half-Contrastive" '
         'Agent=Agents.ExperimentAgent half=true '
         'parallel=true num_workers=20 num_gpus=4 mem=100 '
         'plot_per_steps=0',
         'task=classify/cifar10,classify/tinyimagenet ema=true weight_decay=0.01 '
         'Eyes=Blocks.Architectures.ResNet18 '
         'transform="{RandomHorizontalFlip:{}}" experiment="Third-Label" '
         'Agent=Agents.ExperimentAgent third=true '
         'parallel=true num_workers=20 num_gpus=4 mem=100 '
         'plot_per_steps=0']
# sweep = [f'"experiment=learn-after20k-per1" learn_per_steps=1 '
#          f'num_workers=4 num_gpus=1 mem=20 gpu="K80" '
#          'plot_per_steps=0']
# sweep = [f'experiment=lab-test task=dmc/cheetah_run '
#          f'num_workers=4 num_gpus=1 mem=20 gpu="V100" '
#          'plot_per_steps=0 reservation_id=20220502']

# medium = ['cheetah_run', 'quadruped_walk', 'reacher_easy', 'cup_catch', 'finger_spin', 'walker_walk']
# print(f'CUDA_VISIBLE_DEVICES=0 python Run.py -m Agent=Agents.AC2Agent +num_actors=3 train_steps=500000 seed=1,2,3 task={",".join(["dmc/" + task for task in medium])} experiment="Actor-Ensemble-3"')

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
        s.sendline(f'python sbatch_multirun.py -m {hyperparams} username="{username}" conda="{conda}"')
        s.prompt()
        print(s.before)
    s.logout()
except pxssh.ExceptionPxssh as e:
    print("pxssh failed on login.")
    print(e)
