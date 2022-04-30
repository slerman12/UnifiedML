# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import os
from pathlib import Path

from pexpect import pxssh, spawn

username = 'slerman'
host = 'cornea'
conda = 'conda activate agi'
steps = 500000

# experiments = ["No-Contrastive", "Half-Half-Contrastive", "Third-Label"]
experiments = ["Actor-Ensemble-3", "Actor-Ensemble-5", "Reference"]

# Plot experiments
try:
    s = pxssh.pxssh()
    s.login(host, username)
    s.sendline(f'cd UnifiedML')                         # Run a command
    s.prompt(timeout=None)                              # Match the prompt
    print(s.before)                                     # Print everything before the prompt.
    s.sendline('git pull origin master')
    s.prompt(timeout=None)
    print(s.before)
    s.sendline(conda)
    s.prompt(timeout=None)
    print(s.before)
    print(f'python Plot.py ' + ' '.join(experiments) + (f" steps={steps}" if steps else ""))
    s.sendline(f'python Plot.py ' + ' '.join(experiments) + (f" steps={steps}" if steps else ""))
    s.prompt(timeout=None)
    print(s.before)
    s.logout()
except pxssh.ExceptionPxssh as e:
    print("pxssh failed on login.")
    print(e)

# SFTP experiment results

local_path = f"./Benchmarking/Results/{'_'.join(experiments)}"
remote_path = f"./Benchmarking/{'_'.join(experiments)}/Plots"

Path(local_path).mkdir(parents=True, exist_ok=True)
os.chdir(local_path)

p = spawn(f'sftp {host}')
p.expect('sftp> ')
p.sendline(f"lcd {local_path}")
p.expect('sftp> ')
p.sendline("cd UnifiedML")
p.expect('sftp> ')
p.sendline(f"get {remote_path}/*png")
p.expect('sftp> ')
p.sendline(f"get {remote_path}/*json")
p.expect('sftp> ')
for experiment in experiments:
    p.sendline(f"get -r ./Benchmarking/{experiment}")
    p.expect('sftp> ')

