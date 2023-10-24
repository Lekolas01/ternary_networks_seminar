import torch.nn as nn

from .ternary import TernaryLinear, TernaryModule


class MushroomNet(nn.Module):
    def __init__(self, in_features, a: float, b: float):
        super().__init__()
        self.in_features = in_features
        self.a = a
        self.b = b
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=80, bias=True),
            nn.Linear(in_features=80, out_features=1, bias=True),
            nn.Sigmoid(),
        )


class TernaryMushroomNet(TernaryModule):
    def __init__(self, in_features: int, a: float, b: float, **kwargs):
        classifier = nn.Sequential(
            TernaryLinear(in_features=in_features, out_features=10, bias=True),
            TernaryLinear(in_features=10, out_features=1, bias=True),
            nn.Sigmoid(),
        )
        super().__init__(classifier, a, b, **kwargs)
