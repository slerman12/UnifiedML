import torch
from torch.utils.data import Dataset

from torchaudio.transforms import Spectrogram

from torchvision.transforms import ToPILImage

import numpy as np


class RRUFF(Dataset):
    def __init__(self, root='../XRDs/icsd_Datasets/', data='icsd_171k_ps3', transform=None, num_classes=7, train=True,
                 **kwargs):
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

        # self.spectrogram = Spectrogram()
        # self.image = ToPILImage()
        # self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.FloatTensor(list(map(float, self.features[idx].strip().split(','))))
        y = np.array(list(map(float, self.labels[idx].strip().split(',')))).argmax()

        # x = self.spectrogram(x)
        # x = self.image(x)
        # x = self.transform(x)

        return x, y
