# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import getpass
import os
from pathlib import Path
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

# steps, tasks = None, []

steps = 5e5
experiments = ["Self-Supervised", "DQN-Based", "Reference", "Critic-Ensemble",
               '"linear(2.0,0.1,20000)"', '"linear(2.0,0.1,40000)"']
tasks = ['cheetah_run', 'quadruped_walk', 'reacher_easy', 'cup_catch', 'finger_spin', 'walker_walk',
         'pong', 'breakout', 'boxing', 'krull', 'seaquest', 'qbert']

# Plot experiments
try:
    s = pxssh.pxssh()
    s.login('bluehive.circ.rochester.edu', username, password)
    s.sendline(f'cd /scratch/{username}/UnifiedML')     # Run a command
    s.prompt(timeout=None)                              # Match the prompt
    print(s.before)                                     # Print everything before the prompt.
    s.sendline('git pull origin master')
    s.prompt(timeout=None)
    print(s.before)
    s.sendline(conda)
    s.prompt(timeout=None)
    print(s.before)
    plot_experiments = f"""plot_experiments=['{"','".join(experiments)}']""" if len(experiments) else ""
    plot_tasks = f"""plot_tasks=['{"','".join(tasks)}']""" if len(tasks) else ""
    print(f'python Plot.py {plot_experiments} {plot_tasks} {f"steps={steps}" if steps else ""}')
    s.sendline(f'python Plot.py {plot_experiments} {plot_tasks} {f"steps={steps}" if steps else ""} plot_tabular=true')
    s.prompt(timeout=None)
    print(s.before)
    s.logout()
except pxssh.ExceptionPxssh as e:
    print("pxssh failed on login.")
    print(e)
except pxssh.TIMEOUT as e:
    print("Timeout occurred.")
    print(e)

# SFTP experiment results

local_path = f"./Benchmarking/Results/{'_'.join(experiments)}"
remote_path = f"./Benchmarking/{'_'.join(experiments)}/Plots"

Path(local_path).mkdir(parents=True, exist_ok=True)
os.chdir(local_path)

p = spawn(f'sftp {username}@bluehive.circ.rochester.edu')
p.expect('Password: ', timeout=None)
p.sendline(password)
p.expect('sftp> ', timeout=None)
p.sendline(f"lcd {local_path}")
p.expect('sftp> ', timeout=None)
p.sendline(f"cd /scratch/{username}/UnifiedML")
p.expect('sftp> ', timeout=None)
p.sendline(f"get {remote_path}/*png")
p.expect('sftp> ', timeout=None)
p.sendline(f"get {remote_path}/*json")
p.expect('sftp> ', timeout=None)
for experiment in experiments:
    p.sendline(f"get -r ./Benchmarking/{experiment}")
    p.expect('sftp> ', timeout=None)
