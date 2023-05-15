# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""This file makes it possible to import UnifiedML as a package.

Example:
    In your project, Download the UnifiedML directory:

    $ git clone git@github.com:agi-init/UnifiedML.git

    -------------------------------

    Turn a project file into a UnifiedML-style Run-script that can support all UnifiedML command-line syntax:

    > import UnifiedML   Imports UnifiedML
    >
    > if __name__ == '__main__':
    >    UnifiedML.launch()  # Launches UnifiedML

    -------------------------------

    Say your file is called MyRunner.py and includes an architecture called MyEyes as a class. You can run:

        $ python MyRunner.py Eyes=MyRunner.MyEyes

    or even define your own recipe MyRecipe.yaml in your app's local Hyperparams/task/ directory:

        $ python MyRunner.py task=MyRecipe

    You could also specify hyperparams in-code:

        e.g. directly pass a class to the launcher as such:

        > if __name__ == '__main__':
        >    UnifiedML.launch(Eyes=MyEyes)

        or specify a default recipe:

        > if __name__ == '__main__':
        >    UnifiedML.launch(task='MyRecipe')

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
def launch(**kwargs):
    command_line_args = {arg.split('=')[0] for arg in sys.argv if '=' in arg}
    original = sys.argv
    added = set()

    for key, value in kwargs.items():
        if isinstance(value, (str, bool)):
            if key not in command_line_args:
                sys.argv.append(f'{key}={value}')
                added.add(key)

    global launch_args
    launch_args.update({key: kwargs[key] for key in kwargs.keys() - command_line_args - added})

    from .Run import main
    main()
    sys.argv = original
