import torch.nn as nn

from .ternary import TernaryModule, TernaryLinear


class AdultNet(nn.Module):
    def __init__(self, in_features, **kwargs):
        super().__init__(**kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=80),
            nn.ReLU(),
            nn.Linear(in_features=80, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        ans = self.classifier(x).flatten()
        return ans, ans


class TernaryAdultNet(TernaryModule):
    def __init__(self, in_features: int, a: float, b: float, **kwargs):
        classifier = nn.Sequential(
            TernaryLinear(in_features=in_features, out_features=80),
            nn.Sigmoid(),
            TernaryLinear(in_features=80, out_features=1),
            nn.Sigmoid()
        )
        super().__init__(classifier, a, b, **kwargs)

