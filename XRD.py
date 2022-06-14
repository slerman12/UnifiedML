from torch import nn
import torch

class Model(nn.Module):
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
            # nn.Flatten(1),
            # nn.Linear(1856, 512),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(256, num_classes),
        )

    def forward(self, obs):
        return self.model(obs)