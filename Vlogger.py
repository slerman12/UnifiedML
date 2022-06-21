# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from pathlib import Path
import imageio  # M1 Mac: comment out freeimage imports in imageio/plugins/_init_

import torch
from torchvision.utils import save_image

import wandb


class Vlogger:
    def __init__(self, fps, path='.', reel=False, wandb=False):
        self.save_path = Path(path.replace('Agents.', ''))
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.fps = fps

        # Saves image reels instead of video
        self.reel = reel

        self.wandb = wandb

    def dump_vlogs(self, vlogs, name="Video_Image"):
        if self.reel:
            c, h, w = (min(vlogs[0].shape[-3], 3),  # Undoing frame-stack if necessary (max = 3 channels per image)
                       vlogs[0].shape[-2], vlogs[0].shape[-1])
            vlogs = torch.stack(vlogs).view(-1, c, h, w)
            save_image(vlogs, str(self.save_path / (name + '.png')))
        else:
            # Assumes channel-last format
            imageio.mimsave(str(self.save_path / (name + '.mp4')), vlogs, fps=self.fps)
        if self.wandb:
            wandb.log({"video": wandb.Video(vlogs, fps=self.fps, format="gif")})
