import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import TernaryLinear, TernaryConv2d, R


class LeNet(nn.Module):
    def __init__(self, n_classes):
        super(LeNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


class TernaryLeNet(nn.Module):
    def __init__(self, n_classes: int, a: float = 0.0, b: float = 0.0):
        super(TernaryLeNet, self).__init__()
        self.a = a
        self.b = b

        self.feature_extractor = nn.Sequential(
            TernaryConv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2),
            TernaryConv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2),
            TernaryConv2d(in_channels=64, out_channels=120, kernel_size=5, stride=1),
        )

        self.classifier = nn.Sequential(
            TernaryLinear(in_features=120, out_features=84),
            nn.Dropout(p=0.5),
            TernaryLinear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def regularization(self):
        reg = torch.zeros(1)
        for param in self.parameters():
            reg = reg + R(param, self.a)
        return self.b * reg
