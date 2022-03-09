# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import math
import warnings
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.transforms import transforms, InterpolationMode, functional as vF

import Utils
from Datasets.ReplayBuffer.Classify._TinyImageNet import TinyImageNet


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        # Operates on last 3 dims of x, preserves leading dims
        shape = x.shape
        x = x.view(-1, *shape[-3:])
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        output = F.grid_sample(x,
                               grid,
                               padding_mode='zeros',
                               align_corners=False)
        return output.view(*shape[:-3], *output.shape[-3:])


class IntensityAug(nn.Module):
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        noise = 1.0 + (self.scale * torch.randn(
            (x.shape[0], 1, 1, 1), device=x.device).clamp_(-2.0, 2.0))
        return x * noise


class ComposeAugs(nn.Module):
    def __init__(self, augs):
        super().__init__()
        # Encoder currently expects values in [0, 255], so this is a bit of a workaround for normalization in classify
        if 'Normalize' in augs and 'dataset' in augs['Normalize']:
            for dataset in list(torchvision.datasets.__all__) + ['TinyImageNet']:
                if dataset.lower() == augs['Normalize']['dataset'].lower():
                    break
            assert dataset in torchvision.datasets.__all__ or dataset == 'TinyImageNet'

            path = f'./Datasets/ReplayBuffer/Classify/{dataset}'
            dataset = TinyImageNet if dataset == 'TinyImageNet' else getattr(torchvision.datasets, dataset)

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '.*The given NumPy array.*')
                experiences = dataset(root=path + "_Train", transform=transforms.ToTensor())

            mean, stddev = Utils.data_mean_std(experiences, scale=255)
            del augs['Normalize']['dataset']
            augs['Normalize']['mean'] = mean
            augs['Normalize']['std'] = stddev / 255  # Since Encoder divides pixels by 255

        self.transform = transforms.Compose([getattr(transforms, aug)(**augs[aug]) if hasattr(transforms, aug) else
                                             globals()[aug](**augs[aug]) for aug in augs])

    def forward(self, x):
        return self.transform(x)


class RandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
            self,
            num_ops: int = 2,
            magnitude: int = 9,
            num_magnitude_bins: int = 31,
            interpolation: InterpolationMode = InterpolationMode.NEAREST,
            fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = img.shape[-3:]
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


def _apply_op(
        img: torch.Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    img = img.to(torch.uint8)
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        img = vF.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        img = vF.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
        )
    elif op_name == "TranslateX":
        img = vF.affine(
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        img = vF.affine(
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        img = vF.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = vF.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = vF.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = vF.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = vF.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = vF.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = vF.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = vF.autocontrast(img)
    elif op_name == "Equalize":
        img = vF.equalize(img)
    elif op_name == "Invert":
        img = vF.invert(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError(f"The provided operator {op_name} is not recognized.")
    return img
