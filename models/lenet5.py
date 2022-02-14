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


class TernaryLinear(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        module = nn.Linear(**kwargs)
        self.weight = module.weight
        self.bias = module.bias

        
    def forward(self, x: torch.Tensor):
        weight = torch.tanh(self.weight)
        bias = None if self.bias is None else torch.tanh(self.bias)
        return F.linear(x, weight, bias)


class TernaryConv2d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        module = nn.Conv2d(**kwargs)
        self.weight = module.weight
        self.bias = module.bias
    
    def forward(self, x: torch.Tensor):
        weight = torch.tanh(self.weight)
        bias = None if self.bias is None else torch.tanh(self.bias)
        return F.conv2d(x, weight, bias)


class TernaryLeNet5(nn.Module):
    def __init__(self, n_classes):
        super(TernaryLeNet5, self).__init__()
        
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


class PaperLeNet5(nn.Module):
    def __init__(self, n_classes):
        super(TernaryLeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            TernaryConv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            TernaryConv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            TernaryConv2d(in_channels=64, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            TernaryLinear(in_features=120, out_features=84),
            nn.Tanh(),
            TernaryLinear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
