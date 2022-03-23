import torch.nn as nn

from .ternary import TernaryModule


class AdultNet(nn.Module):
    def __init__(self, in_features=100, **kwargs):
        super().__init__(**kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=80),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.classifier(x).flatten()


class TernaryAdultNet(TernaryModule):
    def __init__(self, in_features: int, a: float, b: float, **kwargs):
        classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=80),
            nn.Linear(in_features=80, out_features=1),
            nn.Sigmoid()
        )
        super().__init__(classifier, a, b, **kwargs)

