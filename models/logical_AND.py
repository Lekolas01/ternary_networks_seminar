import torch.nn as nn

from .ternary import TernaryModule, TernaryLinear


class ANDNet(nn.Module):
    def __init__(self, in_features, **kwargs):
        super().__init__(**kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=1)
        )

    def forward(self, x):
        ans = self.classifier(x).flatten()
        return ans, ans

    def __str__(self) -> str:
        ans = ""
        layers = list(self.classifier)
        for layer in list(self.classifier):
            ans += str(layer.weight)
            ans += str(layer.bias)
        return ans


class TernaryANDNet(TernaryModule):
    def __init__(self, in_features: int, a: float, b: float, **kwargs):
        classifier = nn.Sequential(
            TernaryLinear(in_features=in_features, out_features=1)
        )
        super().__init__(classifier, a, b, **kwargs)
