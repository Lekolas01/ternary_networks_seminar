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
        self.module = nn.Linear(**kwargs)
        
    
    def forward(self, x: torch.Tensor):
        w = torch.tanh(self.module.weight)
        b = None
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            b = torch.tanh(self.module.bias)
        return F.linear(input=x, weight=w, bias=b)


class TernaryConv2d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.module = nn.Conv2d(**kwargs)
    
    def forward(self, x: torch.Tensor):
        w = torch.tanh(self.module.weight)
        b = None
        if hasattr(self.module, 'bias') and self.module.bias is not None:
            b = torch.tanh(self.module.bias)
        return F.conv2d(input=x, weight=w, bias=b)


class TernaryLeNet5(nn.Module):
    def __init__(self, n_classes):
        super(TernaryLeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            TernaryConv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, bias=False),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            TernaryConv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            TernaryConv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            TernaryLinear(in_features=120, out_features=84, bias=False),
            nn.Tanh(),
            TernaryLinear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

