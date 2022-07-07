import time

import numpy as np

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from Blocks.Encoders import adapt_cnn
from Datasets import ExperienceReplay

import Utils


class Encoder(nn.Module):
    def __init__(self, input_shape=1):
        super().__init__()

        in_channels = input_shape if isinstance(input_shape, int) else input_shape[0]

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, stride=1),
            nn.BatchNorm2d(8),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 8, kernel_size=5, stride=1),
            nn.BatchNorm2d(8),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=5, stride=1),
            nn.BatchNorm2d(16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=5, stride=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1),
            torch.nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, obs):
        return self.model(obs)


class Actor(nn.Module):
    def __init__(self, input_shape=1856, output_dim=7):
        super().__init__()

        in_channels = input_shape if isinstance(input_shape, int) else input_shape[0]

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, obs):
        return self.model(obs)


# class Encoder(nn.Module):
#     def __init__(self, input_shape=1):
#         super().__init__()
#
#         in_channels = input_shape if isinstance(input_shape, int) else input_shape[0]
#
#         self.model = nn.Sequential(
#             nn.Conv2d(in_channels, 8, kernel_size=(5, 1), stride=1),
#             nn.BatchNorm2d(8),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=2),
#             nn.Conv2d(8, 8, kernel_size=(5, 1), stride=1),
#             nn.BatchNorm2d(8),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=2),
#             nn.Conv2d(8, 16, kernel_size=(5, 1), stride=1),
#             nn.BatchNorm2d(16),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=2),
#             nn.Conv2d(16, 16, kernel_size=(5, 1), stride=1),
#             nn.BatchNorm2d(16),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=2),
#             nn.Conv2d(16, 32, kernel_size=(5, 1), stride=1),
#             nn.BatchNorm2d(32),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=2),
#             nn.Conv2d(32, 32, kernel_size=(5, 1), stride=1),
#             nn.BatchNorm2d(32),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=2),
#             nn.Conv2d(32, 64, kernel_size=(5, 1), stride=1),
#             nn.BatchNorm2d(64),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=2),
#             nn.Conv2d(64, 64, kernel_size=(5, 1), stride=1),
#             torch.nn.BatchNorm2d(64),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2, 1), stride=2),
#         )
#
#     def forward(self, obs):
#         return self.model(obs)


class Model(nn.Module):
    def __init__(self, model=None):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=1),
            nn.BatchNorm1d(8),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 8, kernel_size=5, stride=1),
            nn.BatchNorm1d(8),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 16, kernel_size=5, stride=1),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 16, kernel_size=5, stride=1),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 32, kernel_size=5, stride=1),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 64, kernel_size=5, stride=1),
            torch.nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(1),
            nn.Linear(1856, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 7),
        ) if model is None \
            else model

        self.optim = SGD(self.model.parameters(), lr=0.001)
        self.scheduler = None

    def forward(self, obs):
        return self.model(obs)


class XRDData(Dataset):
    def __init__(self, root='../XRDs/icsd_Datasets/icsd171k_mix/', train=True, train_eval_split=0.9,
                 num_classes=7, seed=0, transform=None, **kwargs):

        features_path = root + "features.csv"
        label_path = root + f"labels{num_classes}.csv"

        self.classes = list(range(num_classes))

        # Store on CPU
        with open(features_path, "r") as f:
            self.features = f.readlines()
        with open(label_path, "r") as f:
            self.labels = f.readlines()
            full_size = len(self.labels)

        train_size = round(full_size * train_eval_split)

        # Each worker shares an indexing scheme
        rng = np.random.default_rng(seed)
        train_indices = rng.choice(np.arange(full_size), size=train_size, replace=False)
        eval_indices = np.array([idx for idx in np.arange(full_size) if idx not in train_indices])

        self.indices = train_indices if train else eval_indices

        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]

        x = np.array(list(map(float, self.features[idx].strip().split(','))))
        y = np.array(list(map(float, self.labels[idx].strip().split(',')))).argmax()

        if self.transform is not None:
            x = self.transform(x)

        return x, y


# Allows stat aggregations on batches of different lengths
def size_agnostic_agg(stats, agg):
    masked = np.ma.empty((len(stats), max(map(len, stats))))
    masked.mask = True
    for m, stat in zip(masked, stats):
        m[:len(stat)] = stat
    return agg(masked)


if __name__ == '__main__':

    Utils.set_seeds(0)

    twoD = True

    model = Model(nn.Sequential(Encoder(), Actor()) if twoD
                  else None)

    if twoD:
        adapt_cnn(model, (1, 8500, 1))

    # env = Environment('Custom.XRDSynthetic_Dataset', 0, 0, 1000, 1000,
    #                   OmegaConf.create({'_target_': 'Datasets.ReplayBuffer.Classify._XRD.XRDSynthetic'}),
    #                   train=False, suite='classify',
    #                   offline=True, batch_size=16)

    # encoder = CNNEncoder([1, 8500, 1], lr=.001, eyes=Encoder(), optim=torch.optim.SGD)

    # actor = EnsembleGaussianActor(encoder.repr_shape, 50, 1024, (7,), trunk=nn.Identity, pi_head=Actor(),
    #                               ensemble_size=1, optim=torch.optim.SGD, lr=.001)

    # agent = DQNAgent([1, 8500, 1], (7,), 50, 1024, [float('nan')] * 4, False, False,
    #                  OmegaConf.create({'encoder': {'eyes': {'_target_': 'XRD.Encoder'}},
    #                                    'actor': {'trunk': {'_target_': 'Blocks.Architectures.Null'},
    #                                              'pi_head': {'_target_': 'XRD.Actor'}},
    #                                    'critic': {}, 'aug': {'_target_': 'Blocks.Architectures.Null'}}),
    #                  0.001, 0, 0, 0, False, 0, 1, torch.inf, False, False, True, False, 'cpu', False, True)

    epochs = 50
    log_interval = 1000
    batch_size = 16
    train_test_split = 0.9

    print("parsing train...")
    train_loader = ExperienceReplay(batch_size, 1, np.inf, {'name': 'action', 'shape': (7,)}, 'classify',
                                    'Custom.XRDSynthetic_Dataset', True, False, False, False, False,
                                    '.Datasets/ReplayBuffer/Classify/Custom.XRDSynthetic_Dataset',
                                    {'name': 'observation', 'shape': (8500,)})
    print("done, size", len(train_loader))

    print("parsing test...")
    test_dataset = XRDData(train=False, train_eval_split=train_test_split)
    eval_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("done, size", len(eval_loader))

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        accuracy = []
        for i in range(len(train_loader)):
            obs, _, _, _, _, label, *_ = Utils.to_torch(next(train_loader), 'cpu')
            if not twoD:
                obs = obs.squeeze(-1)
            y_predicted = model(obs)
            supervised_loss = cross_entropy(y_predicted, label.long())
            Utils.optimize(supervised_loss, model)

            accuracy.append((torch.argmax(y_predicted, -1) == label).float().numpy())

            if i and i % log_interval == 0:
                print(time.time() - start_time, size_agnostic_agg(accuracy, np.ma.mean))

        accuracy = []
        for obs, label in eval_loader:
            with torch.no_grad():
                obs = obs.float()
                obs = obs.unsqueeze(1)
                if twoD:
                    obs = obs.unsqueeze(-1)
                y_predicted = model(obs)
                accuracy.append((y_predicted.argmax(-1) == label).float().numpy())

        print(epoch)
        print('Eval accuracy', size_agnostic_agg(accuracy, np.ma.mean))
        print(time.time() - start_time)
