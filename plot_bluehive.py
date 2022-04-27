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

experiments = ["'linear(1.0,0.1,20000)'", "'linear(0.5,0.1,20000)'",
               "'linear(1.0,0.4,20000)'", "'linear(1.0,0.1,40000)'"]

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
    s.sendline('python Plot.py ' + ' '.join(experiments))
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

p = spawn(f'sftp {username}@bluehive.circ.rochester.edu')
p.expect('Password: ')
p.sendline(password)
p.expect('sftp> ')
p.sendline(f"lcd {local_path}")
p.expect('sftp> ')
p.sendline(f"cd /scratch/{username}/UnifiedML")
p.expect('sftp> ')
p.sendline(f"get {remote_path}/*png")
p.expect('sftp> ')
p.sendline(f"get {remote_path}/*json")
p.expect('sftp> ')
for experiment in experiments:
    p.sendline(f"get -r ./Benchmarking/{experiment}")
    p.expect('sftp> ')

