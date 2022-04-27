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

# Define sweep
# sweep = ['task=classify/cifar10,classify/tinyimagenet ema=true weight_decay=0.01 '
#          'Eyes=Blocks.Architectures.ResNet18 '
#          'transform="{RandomHorizontalFlip:{}}" experiment="Supervised-RL" '
#          'parallel=true num_workers=20 num_gpus=4 mem=100 '
#          'plot_per_steps=0']
sweep = ['task=dmc/cheetah_run,atari/pong,classify/mnist experiment="agi_A100_test" train_steps=100000 '
         'num_workers=4 num_gpus=1 mem=20 gpu="A100" '
         'plot_per_steps=0']

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
        s.sendline(f'python sbatch_multirun.py -m {hyperparams} conda="{conda}"')
        s.prompt()
        print(s.before)
    s.logout()
except pxssh.ExceptionPxssh as e:
    print("pxssh failed on login.")
    print(e)
