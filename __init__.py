# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""
This file makes it possible to import UnifiedML as a package.
"""

import sys
import os
import inspect

UnifiedML = os.path.dirname(__file__)
app = '/'.join(str(inspect.stack()[-1][1]).split('/')[:-1])


# Imports UnifiedML paths and the paths of any launching app
def import_paths():
    sys.path.extend([UnifiedML, app])  # Imports UnifiedML paths and the paths of the launching app

    if os.path.exists(app + '/Hyperparams'):
        sys.argv.extend(['-cd', app + '/Hyperparams'])  # Adds an app's Hyperparams dir to Hydra's .yaml search path


import_paths()


launch_args = {}


# Launches UnifiedML from inside a launching app with specified args
def launch(**hyperparams):
    original = list(sys.argv)

    command_line_args = {arg.split('=')[0] for arg in sys.argv if '=' in arg}
    added = set()

    for key, value in hyperparams.items():
        if isinstance(value, (str, bool)):
            if key not in command_line_args:
                sys.argv.insert(-2, f'{key}={value}')  # For Hydra registered resolvers in Utils
                added.add(key)

    global launch_args
    launch_args = {key: hyperparams[key] for key in hyperparams.keys() - command_line_args - added}

    from .Run import main
    main()  # Run

    launch_args = {}
    sys.argv = original
