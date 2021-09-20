import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.is_training = True

        self._conv0 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=0)

        self._conv1 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=2)
        self._bn1 = nn.BatchNorm2d(32)
        self._rely1 = nn.LeakyReLU(inplace=True)

        self._conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=2)
        self._bn2 = nn.BatchNorm2d(64)
        self._rely2 = nn.LeakyReLU(inplace=True)

        self._conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=2)
        self._bn3 = nn.BatchNorm2d(128)
        self._rely3 = nn.LeakyReLU(inplace=True)

        self._conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=2, padding=2)
        self._bn4 = nn.BatchNorm2d(256)
        self._rely4 = nn.LeakyReLU(inplace=True)

        self._flatten = nn.Flatten()
        self._linear1 = nn.Linear(2304, 256)
        self._linear2 = nn.Linear(256, 2)

        self._backbone = nn.ModuleList([
            self._conv0,

            self._conv1,
            self._bn1,
            self._rely1,

            self._conv2,
            self._bn2,
            self._rely2,

            self._conv3,
            self._bn3,
            self._rely3,

            self._conv4,
            self._bn4,
            self._rely4,

            self._flatten,
            self._linear1,
            self._linear2
        ])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, module in enumerate(self._backbone):
            x = module(x)

        if not self.is_training:
            x = self.softmax(x)

        return x
