# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import getpass
import os
from cryptography.fernet import Fernet

from pexpect import spawn


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

p = spawn('/opt/cisco/anyconnect/bin/vpn connect vpnconnect.rochester.edu')
p.expect('Username: ')
p.sendline('')
p.expect('Password: ')
p.sendline(password)
p.expect('Second Password: ')
p.sendline('push')
p.expect('VPN>')

# To disconnect:
# p = spawn('/opt/cisco/anyconnect/bin/vpn disconnect')
# p.expect('b')

print('Connected to VPN\nFor Bluehive:\nssh slerman@bluehive.circ.rochester.edu')
