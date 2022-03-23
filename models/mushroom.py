import torch.nn as nn

from .ternary import TernaryLinear


class MushroomNet(nn.Module):
    def __init__(self, in_features, a: float, b: float):
        super().__init__()
        self.in_features = in_features
        self.a = a
        self.b = b
        self.classifier = nn.Sequential(
            TernaryLinear(in_features=in_features, out_features=80, bias=True),
            TernaryLinear(in_features=80, out_features=1, bias=True),
            nn.Sigmoid()
        )

