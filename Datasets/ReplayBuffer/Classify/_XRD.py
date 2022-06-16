import torch
from torch.utils.data import Dataset

from torchaudio.transforms import Spectrogram

from torchvision.transforms import ToPILImage

import numpy as np


class RRUFF(Dataset):
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


class Synthetic(Dataset):
    def __init__(self, root='../XRDs/icsd_Datasets/', data='icsd171k_mix', transform=None, num_classes=7, train=True,
                 train_test_split=0.9, **kwargs):

        self.feature_path = root + data + "/features.csv"
        self.label_path = root + data + f"/labels{num_classes}.csv"

        self.classes = list(range(num_classes))

        with open(self.feature_path, "r") as f:
            self.feature_lines = f.readlines()
        with open(self.label_path, "r") as f:
            self.label_lines = f.readlines()

        self.num_datapoints = len(self.label_lines)

        self.train_test_split = train_test_split
        self.size = train_size = round(self.num_datapoints * self.train_test_split)
        self.train = train
        if not self.train:
            self.size = self.num_datapoints - train_size

        self.train_inds = np.random.choice(np.arange(self.num_datapoints), size=train_size, replace=False)
        self.test_inds = np.array([x for x in np.arange(self.num_datapoints) if x not in self.train_inds])

        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.train:
            idx = self.train_inds[idx]
        else:
            idx = self.test_inds[idx]
        line = self.feature_lines[idx]
        x = list(map(float, line.strip().split(",")))
        x = torch.FloatTensor(x)
        x = self.transform(x)
        line = self.label_lines[idx]
        y = list(map(float, line.strip().split(",")))
        y = torch.FloatTensor(y)
        y = torch.argmax(y)
        return x, y
