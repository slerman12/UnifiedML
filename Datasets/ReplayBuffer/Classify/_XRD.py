# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import numpy as np

from torch.utils.data import Dataset

from torchvision.transforms import ToPILImage

from torchaudio.transforms import Spectrogram


class XRDRRUFF(Dataset):
    def __init__(self, root='../XRDs/icsd_Datasets/', data='icsd_171k_ps3', transform=None, num_classes=7, train=True,
                 spectrogram=False, **kwargs):
        root += data if train else 'rruff/XY_DIF_noiseAll'

        self.feature_path = root + "/features.csv"
        self.label_path = root + f"/labels{num_classes}.csv"

        self.classes = list(range(num_classes))

        with open(self.feature_path) as f:
            self.features = f.readlines()
        with open(self.label_path) as f:
            self.labels = f.readlines()

        self.size = len(self.features)
        assert self.size == len(self.labels), 'num features and labels not same'

        if spectrogram:
            self.spectrogram = Spectrogram()
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = np.array(list(map(float, self.features[idx].strip().split(','))))
        y = np.array(list(map(float, self.labels[idx].strip().split(',')))).argmax()

        x = self.transform(x)

        if hasattr(self, 'spectrogram'):
            x = self.spectrogram(x)

        return x, y


class XRDSynthetic(Dataset):
    def __init__(self, root='../XRDs/icsd_Datasets/icsd171k_mix/', train=True, train_test_split=0.9,
                 num_classes=7, use_cpu_memory=True, transform=None, seed=0, **kwargs):

        self.features_path = root + "features.csv"
        self.label_path = root + f"labels{num_classes}.csv"

        self.use_cpu_memory = use_cpu_memory

        full_size = sum(1 for _ in open(self.features_path, 'rb'))  # Memory-efficient counting
        train_size = round(full_size * train_test_split)

        # Each worker shares an indexing scheme
        rng = np.random.default_rng(seed)

        # Train indices
        self.indices = rng.choice(np.arange(full_size), size=train_size, replace=False)

        # Eval indices
        if not train:
            self.indices = np.array([idx for idx in np.arange(full_size) if idx not in self.indices])

        self.classes = list(range(num_classes))

        # Can load slowly from disk with low memory footprint or CPU
        if self.use_cpu_memory:
            # Store on CPU
            with open(self.features_path, "r") as f:
                self.features_lines = f.readlines()
            with open(self.label_path, "r") as f:
                self.label_lines = f.readlines()
        else:
            self.features_lines = self.label_lines = None

        self.transform = transform

    def __len__(self):
        return len(self.indices)

    # Get a line from a file in disk or stored on CPU
    def get_line(self, idx, name):
        if self.use_cpu_memory:
            return getattr(self, name + '_lines')[idx]
        else:
            with open(getattr(self, name + '_path'), "r") as f:
                for i, line in enumerate(f):
                    if i == self.indices[idx]:
                        return line

    def __getitem__(self, idx):
        idx = self.indices[idx]

        # Features and label
        x = np.array(list(map(float, self.get_line(idx, 'features').strip().split(','))))
        y = np.array(list(map(float, self.get_line(idx, 'label').strip().split(',')))).argmax()

        # Data transforms
        if self.transform is not None:
            x = self.transform(x)

        return x, y
