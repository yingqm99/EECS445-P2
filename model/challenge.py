'''
EECS 445 - Introduction to Machine Learning
Fall 2020 - Project 2
Challenge
    Constructs a pytorch model for a convolutional neural network
    Usage: from model.challenge import Challenge
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
class Challenge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv4 = nn.Conv2d(128, 32, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)
        # TODO:

        #

        self.init_weights()

    def init_weights(self):
        # TODO:
        for conv in [self.conv1, self.conv2, self.conv3]:
            C_in = conv.weight.size(1)
            # 怎么Normal distribute的？
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5*5*C_in))
            nn.init.constant_(conv.bias, 0.0)

        for fc in [self.fc1, self.fc2, self.fc3]:
            C_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(C_in))
            nn.init.constant_(fc.bias, 0.0)

        #

    def forward(self, x):
        N, C, H, W = x.shape
        dropout = nn.Dropout(p=0.05)
        x = dropout(F.relu(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        # x = self.pool2(x)
        # x = dropout(F.relu(self.conv3(x)))
        # x = self.pool3(x)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.Tensor(x)
        # TODO:

        #

        # return z
