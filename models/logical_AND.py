import torch.nn as nn

from .ternary import TernaryModule, TernaryLinear


class ANDNet(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())

    def forward(self, x):
        ans = self.classifier(x).flatten()
        return ans


class TernaryANDNet(TernaryModule):
    def __init__(self, in_features: int, a: float, b: float, **kwargs):
        classifier = nn.Sequential(
            TernaryLinear(in_features=in_features, out_features=1),
        )
        super().__init__(classifier, a, b, **kwargs)
