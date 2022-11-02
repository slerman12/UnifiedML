# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import getpass
import os
from cryptography.fernet import Fernet

from pexpect import pxssh, spawn

from Sweeps.SweepsAndPlots import runs


username = 'slerman'

branch = 'OldStats'  # Note: Perhaps specify in sweeps_and_plots.runs instead

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

conda = f'source /home/{username}/miniconda3/bin/activate CUDA11.3'

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

sweep_group = list(runs.keys())[0]
sweep = runs[sweep_group].sweep


# Launch on Bluehive
try:
    s = pxssh.pxssh()
    s.login('bluehive.circ.rochester.edu', username, password)
    s.sendline(f'cd /scratch/{username}/UnifiedML')     # Run a command
    s.prompt()                                          # Match the prompt
    print(s.before.decode("utf-8"))                     # Print everything before the prompt.
    s.sendline(f'git fetch origin')
    s.prompt()
    print(s.before.decode("utf-8"))
    s.sendline(f'git checkout -b {branch} origin/{branch or "master"}')
    s.prompt()
    prompt = s.before.decode("utf-8")
    if f"fatal: A branch named '{branch}' already exists." in prompt:
        s.sendline(f'git checkout {branch}')
        s.prompt()
        prompt = s.before.decode("utf-8")
    print(prompt)
    assert 'error' not in prompt
    s.sendline(f'git pull origin {branch}')
    s.prompt()
    print(s.before.decode("utf-8"))
    s.sendline(conda)
    s.prompt()
    print(s.before.decode("utf-8"))
    for i, hyperparams in enumerate(sweep):
        hyperparams = "\t".join(hyperparams.splitlines())
        print(f'Set: {i + 1}')
        print(f'python sbatch.py -m {hyperparams}   username="{username}"\n')
        s.sendline(f'python sbatch.py -m {hyperparams} username="{username}"')
        s.prompt()
        print(s.before.decode("utf-8"))
    s.logout()
except pxssh.ExceptionPxssh as e:
    print("pxssh failed on login.")
    print(e)
