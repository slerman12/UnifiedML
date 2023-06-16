# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
"""
This file makes it possible to import UnifiedML as a package.
"""

import os
import inspect

from Utils import launch
from Hyperparams.minihydra import yaml_search_paths

UnifiedML = os.path.dirname(__file__)
app = '/'.join(str(inspect.stack()[-1][1]).split('/')[:-1])


# Imports UnifiedML paths and the paths of any launching app
def import_paths():
    yaml_search_paths.extend([UnifiedML, app])  # Imports UnifiedML paths and the paths of the launching app

    if os.path.exists(app + '/Hyperparams'):
        yaml_search_paths.append(app + '/Hyperparams')  # Adds an app's Hyperparams dir to search path


import_paths()
