# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""This file makes it possible to import UnifiedML as a package.

Example:
    In your project, Download the UnifiedML directory:

    $ git clone git@github.com:agi-init/UnifiedML.git

    -------------------------------

    Turn a project file into a UnifiedML-style Run-script that can support all UnifiedML command-line syntax

    > import sys
    > sys.path.append("./UnifiedML")  # Imports UnifiedML expected syntax and paths

    If you have one, you can add your project's Hyperparams/task/ directory to Hydra's .yaml search path with this line:

    > sys.argv.extend(['-cd', 'Hyperparams'])  # Adds this project's Hyperparams/task/ to Hydra's .yaml search path

    Now you can launch UnifiedML via your project file with access to project-level modules, Datasets, recipes, etc.:

    > from UnifiedML.Run import main   Imports UnifiedML
    >
    > if __name__ == '__main__':
    >    main()  # For launching UnifiedML

    -------------------------------

    Examples:

    Say your file is called MyRunner.py and includes an architecture called MyEyes. You can run:

    $ python MyRunner.py Eyes=MyRunner.MyEyes

    or even define your own recipe MyRecipe.yaml in your local Hyperparams/task/ directory:

    $ python MyRunner.py task=MyRecipe

"""
