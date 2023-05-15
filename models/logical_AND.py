import torch.nn as nn

from .ternary import TernaryModule, TernaryLinear


class ANDNet(nn.Module):
    def __init__(self, in_features, **kwargs):
        super().__init__(**kwargs)
        self.classifier = nn.Linear(in_features=in_features, out_features=1)

    def forward(self, x):
        ans = self.classifier(x).flatten()
        return ans, ans
    
    def __str__(self) -> str:
        ans = f"{self.classifier.weight[0,0].item():.4f}*x1\t"
        if self.classifier.weight[0,1] >= 0:
            ans += '+'
        ans += f"{self.classifier.weight[0,1].item():.4f}*x2\t"
        if self.classifier.bias >= 0:
            ans += '+'
        ans += f"{self.classifier.bias.item():.4f}"
        return ans



class TernaryANDNet(TernaryModule):
    def __init__(self, in_features: int, a: float, b: float, **kwargs):
        classifier = nn.Sequential(
            TernaryLinear(in_features=in_features, out_features=1)
        )
        super().__init__(classifier, a, b, **kwargs)
