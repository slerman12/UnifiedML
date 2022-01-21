# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from pathlib import Path
import imageio  # M1 Mac: comment out freeimage imports in imageio/plugins/_init_

import torch
from torchvision.utils import save_image


class Vlogger:
    def __init__(self, fps, path='.', reel=False):
        self.save_path = Path(path.replace('Agents.', ''))
        self.save_path.mkdir(exist_ok=True, parents=True)
        self.fps = fps

        # Saves image reels instead of video
        self.reel = reel

    def dump_vlogs(self, vlogs, name="Video_Image"):
        if self.reel:
            save_image(torch.stack(vlogs).view(-1, *vlogs[0].shape[:-3]), str(self.save_path / (name + '.png')))
        else:
            imageio.mimsave(str(self.save_path / (name + '.mp4')), vlogs, fps=self.fps)


# Note: May be able to video record more efficiently with:

    # frame = cv2.resize(exp.obs[-3:].transpose(1, 2, 0),
    #                    dsize=(self.render_size, self.render_size),
    #                    interpolation=cv2.INTER_CUBIC)

# in Environment.py
