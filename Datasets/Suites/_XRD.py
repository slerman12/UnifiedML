# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from collections import Iterable
import random

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset

from torchvision.transforms import ToPILImage, Compose

from torchaudio.transforms import Spectrogram


# CPU-memory
class XRD(Dataset):
    def __init__(self, roots=('../XRDs/icsd_Datasets/icsd171k_mix/',), train=True, train_eval_splits=(0.9,),
                 num_classes=7, seed=0, transform=None, spectrogram=False, **kwargs):

        if not isinstance(roots, Iterable):
            roots = (roots,)
        if not isinstance(train_eval_splits, Iterable):
            train_eval_splits = (train_eval_splits,)

        assert len(roots) == len(train_eval_splits), 'Must provide train test split for each root dir'

        self.indices = []
        self.features = {}
        self.labels = {}

        for i, (root, split) in enumerate(zip(roots, train_eval_splits)):
            features_path = root + "features.csv"
            label_path = root + f"labels{num_classes}.csv"

            self.classes = list(range(num_classes))

            print(f'Loading [root={root}, split={split if train else 1 - split}, train={train}] to CPU...')

            # Store on CPU
            with open(features_path, "r") as f:
                self.features[i] = f.readlines()
            with open(label_path, "r") as f:
                self.labels[i] = f.readlines()
                full_size = len(self.labels[i])

            print('Data loaded ✓')

            train_size = round(full_size * split)

            full = range(full_size)

            # Each worker shares an indexing scheme
            random.seed(seed)
            train_indices = random.sample(full, train_size)
            eval_indices = set(full).difference(train_indices)

            indices = train_indices if train else eval_indices
            self.indices += zip([i] * len(indices), list(indices))

        self.transform = transform

        if spectrogram:
            self.spectrogram = Compose([Spectrogram(), ToPILImage()])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        root, idx = self.indices[idx]

        x = torch.FloatTensor(list(map(float, self.features[root][idx].strip().split(','))))
        y = np.array(list(map(float, self.labels[root][idx].strip().split(',')))).argmax()

        if hasattr(self, 'spectrogram'):
            x = self.spectrogram(x)

        # Data transforms
        if self.transform is not None:
            x = self.transform(x)

        return x, y


# File-memory (using linecache)
# https://stackoverflow.com/questions/57138527/can-linecache-be-used-for-concurrent-reading
# from linecache import getline
# class XRD(Dataset):
#     def __init__(self, roots=('../XRDs/icsd_Datasets/icsd171k_mix/',), train=True, train_eval_splits=(0.9,),
#                  num_classes=7, seed=0, transform=None, spectrogram=False, **kwargs):
#         if not isinstance(roots, Iterable):
#             roots = (roots,)
#         if not isinstance(train_eval_splits, Iterable):
#             train_eval_splits = (train_eval_splits,)
#
#         assert len(roots) == len(train_eval_splits), 'must provide train test split for each root dir'
#
#         self.features = [root + 'features.csv' for root in roots]
#         self.labels = [root + f'labels{num_classes}.csv' for root in roots]
#
#         self.classes = list(range(num_classes))
#
#         self.indices = []
#
#         for i, (root, split) in enumerate(zip(roots, train_eval_splits)):
#             print(f'Parsing [root={root}, split={split if train else 1 - split}, train={train}]...')
#
#             full_size = sum(1 for _ in open(self.features[i], 'rb'))  # Memory-efficient counting
#             train_size = round(full_size * split)
#
#             full = range(full_size)
#
#             # Each worker shares an indexing scheme
#             random.seed(seed)
#             train_indices = random.sample(full, train_size)
#             eval_indices = set(full).difference(train_indices)
#
#             indices = train_indices if train else eval_indices
#             self.indices += zip([i] * len(indices), list(indices))
#
#             print('Parsed ✓')
#
#         self.transform = transform
#
#         if spectrogram:
#             self.spectrogram = Compose([Spectrogram(), ToPILImage()])
#
#     def __len__(self):
#         return len(self.indices)
#
#     def __getitem__(self, idx):
#         root, idx = self.indices[idx]
#
#         x = torch.FloatTensor(list(map(float, getline(self.features[root], idx + 1).strip('\n').split(','))))
#         y = np.array(list(map(float, getline(self.labels[root], idx + 1).strip('\n').split(',')))).argmax()
#
#         if hasattr(self, 'spectrogram'):
#             x = self.spectrogram(x)
#
#         # Data transforms
#         if self.transform is not None:
#             x = self.transform(x)
#
#         return x, y
