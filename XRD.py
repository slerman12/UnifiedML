import os
import time
import random

import numpy as np

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

from IPython.display import clear_output

import Utils
from Blocks.Actors import EnsembleGaussianActor
from Blocks.Encoders import adapt_cnn, CNNEncoder
from Datasets.ExperienceReplay import ExperienceReplay


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


config2Theta = "highRes2Theta_5to90"
configTraining = "icsd171k_mix"
num_classes = 7
modelIdx = 11

folder = f"../XRDs/icsd_Datasets/{configTraining}_{config2Theta}/CNN_model_para{modelIdx}/"
if not os.path.exists(folder):
    os.makedirs(folder)
featuresFolder = f"../XRDs/icsd_Datasets/{configTraining}/"


class XRDData(Dataset):
    def __init__(self, train=True, train_test_split=0.9):

        # read feature and label files
        self.feature_file = featuresFolder + "features.csv"
        self.label_file = featuresFolder + f"labels{num_classes}.csv"
        with open(self.feature_file, "r") as f:
            self.feature_lines = f.readlines()
        with open(self.label_file, "r") as f:
            self.label_lines = f.readlines()

        self.num_datapoints = len(self.label_lines)
        print(f"datapoint: {self.num_datapoints}")

        self.train_test_split = train_test_split
        self.size = train_size = round(self.num_datapoints * self.train_test_split)
        self.train = train
        if not self.train:
            self.size = self.num_datapoints - train_size

        self.train_inds = np.random.choice(np.arange(self.num_datapoints), size=train_size, replace=False)
        self.test_inds = np.array([x for x in np.arange(self.num_datapoints) if x not in self.train_inds])

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
        line = self.label_lines[idx]
        y = list(map(float, line.strip().split(",")))
        y = torch.FloatTensor(y)
        y = torch.argmax(y)
        return x, y


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


if __name__ == '__main__':
    # seed = 10
    # torch.manual_seed(seed)
    # random.seed(seed)
    Utils.set_seeds(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = nn.Sequential(
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
        nn.BatchNorm1d(64),
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
        nn.Linear(256, num_classes),
        # nn.Tanh()
    )

    twoD = True
    if twoD:
        model = nn.Sequential(Encoder(), Actor())
        adapt_cnn(model, (1, 1, 8500))
        #
        # model2.load_state_dict(model.state_dict(), strict=True)
        # model = model2

    model.to(device)

    encoder = CNNEncoder([1, 1, 8500], standardize=False, lr=.001, eyes=Encoder(), optim=torch.optim.SGD)
    actor = EnsembleGaussianActor(encoder.repr_shape, 0, 0, (7,), trunk=nn.Identity, pi_head=Actor(), ensemble_size=1, optim=torch.optim.SGD, lr=.001)

    epochs = 50
    log_interval = 1000
    batch_size = 16
    lr = 0.001
    epochLog = folder+f"trainAccLog_classNum{num_classes}.txt"
    #     testsetLog = folder+f"testSetLog_classNum{num_classes}_epochs{epochs}.txt"
    with open(epochLog, "w") as f:
        pass

    train_test_split = 0.9
    print("parsing train...")
    # train_dataset = XRDData(train=True, train_eval_split=train_test_split)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = ExperienceReplay(batch_size, 1, np.inf, {'name': 'action', 'shape': (7,)}, 'classify',
                                    'Custom.XRDSynthetic_Dataset', True, False, False, False, False,
                                    '.Datasets/ReplayBuffer/Classify/Custom.XRDSynthetic_Dataset',
                                    {'name': 'observation', 'shape': (8500,)})
    print("done, size", len(train_loader))
    print("parsing test...")
    test_dataset = XRDData(train=False, train_eval_split=train_test_split)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("done, size", len(test_loader))

    optim = SGD(model.parameters(), lr=lr)
    cost = torch.nn.CrossEntropyLoss()
    loss_stat = correct = total = 0
    start_time = time.time()
    i = 1
    for epoch in range(1, epochs+1):

        # start of epoch
        if (epoch) % 5 == 0:
            torch.save(model, folder+f"model__classNum{num_classes}_epochs{epoch}.pt")

        # training process
        for _ in range(len(train_loader)):
            x, action, reward, discount, next_obs, y, *traj, step, ids, meta = next(train_loader)
            x, y = x.to(device), y.to(device)
            x = x.float()
            x = torch.flatten(x, start_dim=1)
            x = x.unsqueeze(1)
            if twoD:
                x = x.unsqueeze(1)
            y = y.long()
            # y_pred = model(x)
            y_pred = actor.train()(encoder.train()(x)).mean
            loss = cross_entropy(y_pred, y)
            loss_stat += loss.item()
            correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
            total += y.shape[0]
            if i % log_interval == 0:
                pLoss = loss_stat/log_interval
                pAcc = 100.*correct/total
                with open(epochLog, "a") as f:
                    print(f"Train,Epoch:{epoch},Loss:{pLoss:.5f},Acc:{correct}/{total} ({pAcc:.1f}%)", file=f)
                print(f"Train,Epoch:{epoch},Loss:{pLoss:.5f},Acc:{correct}/{total} ({pAcc:.1f}%)")
                loss_stat = correct = total = 0

                print(time.time() - start_time)
            # optim.zero_grad()
            # loss.backward()
            # optim.step()
            Utils.optimize(loss, actor, retain_graph=True)
            Utils.optimize(None, encoder)
            i += 1

        # testing process
        correct = total = 0
        y_pred_all = None
        y_test_all = None

        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
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

        with open(epochLog, "a") as f:
            pAcc = 100.*correct/total
            print(f"Test,Epoch:{epoch},Acc:{correct}/{total} ({pAcc:.1f}%)", file=f)
        print(f"Test,Epoch:{epoch},Acc:{correct}/{total} ({pAcc:.1f}%)")

        # End of epoch
        with open(epochLog, "a") as f:
            print(f"Loop,Epoch:{epoch},Time:{time.time()-start_time:.1f}", file=f)
        print(f"Loop,Epoch:{epoch},Time:{time.time()-start_time:.1f}")
        clear_output(wait=True)
    print("Done")
