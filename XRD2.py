import time

from IPython.display import clear_output

from omegaconf import OmegaConf

import numpy as np

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from Agents import DQNAgent
from Blocks.Actors import EnsembleGaussianActor
from Blocks.Encoders import adapt_cnn, CNNEncoder
from Datasets.Environment import Environment
from Datasets.ExperienceReplay import ExperienceReplay

import Utils


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

    Utils.set_seeds(2)

    twoD = False

    model = Model()

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
    train_dataset = XRDData(train=True, train_test_split=train_test_split)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("done")

    print("parsing test...")
    test_dataset = XRDData(train=False, train_test_split=train_test_split)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print("done")

    optim = SGD(model.parameters(), lr=0.001)
    cost = nn.CrossEntropyLoss()
    loss_stat = correct = total = 0
    start_time = time.time()
    i = 1
    for epoch in range(1, epochs+1):

        model.train()

        # training process
        for x, y in train_loader:
            x, y = x.to('cpu'), y.to('cpu')
            x = x.float()
            x = torch.flatten(x, start_dim=1)
            x = x.unsqueeze(1)
            y_pred = model(x)
            loss = cost(y_pred, y)
            loss_stat += loss.item()
            correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total += y.shape[0]
            if i % log_interval == 0:
                pLoss = loss_stat/log_interval
                pAcc = 100.*correct/total
                print(f"Train,Epoch:{epoch},Loss:{pLoss:.5f},Acc:{correct}/{total} ({pAcc:.1f}%)")
                loss_stat = correct = total = 0
            optim.zero_grad()
            loss.backward()
            optim.step()
            i += 1

        # testing process
        correct = total = 0
        y_pred_all = None
        y_test_all = None

        for i, (x, y) in enumerate(test_loader):
            x, y = x.to('cpu'), y.to('cpu')
            x = x.float()
            x = x.unsqueeze(1)
            modelEval = model.eval()
            y_pred = modelEval(x).detach()
            if epoch == epochs - 1:
                if y_pred_all is None:
                    y_pred_all = y_pred
                else:
                    y_pred_all = torch.cat([y_pred_all, y_pred], dim=0)
                if y_test_all is None:
                    y_test_all = y
                else:
                    y_test_all = torch.cat([y_test_all, y], dim=0)

            correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total += y.shape[0]

        print(f"Test,Epoch:{epoch},Acc:{correct}/{total} ({pAcc:.1f}%)")

        # End of epoch
        print(f"Loop,Epoch:{epoch},Time:{time.time()-start_time:.1f}")
        clear_output(wait=True)
print("Done")
