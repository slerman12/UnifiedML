# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import torch

import Utils

from Agents import DrQV2Agent


class DDPGAgent(DrQV2Agent):
    """
    Deep Deterministic Policy Gradient
    (https://arxiv.org/pdf/1509.02971.pdf)
    """
    def __init__(self, recipes, stddev_schedule, **kwargs):
        recipes.aug = torch.nn.Identity()

        super().__init__(recipes=recipes, stddev_schedule=None, **kwargs)  # Use specified start sched
