import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
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


class TernaryConv2d(nn.Module):
    def __init__(self, **conv2d_args):
        super().__init__()
        conv2d = nn.Conv2d(**conv2d_args)
        self.weight = conv2d.weight
        self.bias = conv2d.bias

    def forward(self, x: torch.Tensor):
        tanh_weight = F.tanh(self.weight)
        tanh_bias = F.tanh(self.bias) if hasattr(self, 'bias') else None
        return F.conv2d(x, tanh_weight, tanh_bias)


class TernaryLinear(nn.Module):
    def __init__(self, **lin_args):
        super().__init__()
        self.linear = nn.Linear(**lin_args)
        self.weight = self.linear.weight
        self.bias = self.linear.bias

    def forward(self, x):
        tanh_weight = torch.tanh(self.linear.weight)
        if (self.linear.bias is not None):
            tanh_bias = torch.tanh(self.linear.bias)
        return F.linear(input=x, weight=tanh_weight, bias=tanh_bias)


class TernaryLeNet5(nn.Module):
    def __init__(self, n_classes):
        super(TernaryLeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.TernaryConv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.TernaryConv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.TernaryConv2d(in_channels=64, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.TernaryLinear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.TernaryLinear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

