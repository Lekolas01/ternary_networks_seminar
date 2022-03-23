import torch
import torch.nn as nn
import torch.nn.functional as F


def R(weights: torch.Tensor, a: float):
    s = torch.tanh(weights) ** 2
    return torch.sum((a - s) * s)


class TernaryModule(nn.Module):
    def __init__(self, classifier: nn.Module, a: float, b: float):
        super().__init__()
        self.classifier = classifier
        self.a = a
        self.b = b

    def forward(self, x):
        ans = self.classifier(x).flatten()
        return ans, ans

    def regularization(self):
        reg = torch.zeros(1).to(self.classifier[0].weight.device)
        for param in self.parameters():
            reg = reg + R(param, self.a)
        return self.b * reg

    def quantized(self):
        seq = self.classifier
        for idx, layer in enumerate(seq):
            layer.zero_grad()
            if isinstance(layer, TernaryLinear):
                new_layer = nn.Linear(in_features=layer.weight.shape[1], out_features=layer.weight.shape[0], bias=True)
                new_layer.requires_grad_(False)
                new_layer.weight.copy_(torch.round(torch.tanh(layer.weight.detach())))
                new_layer.bias.copy_(torch.round(torch.tanh(layer.bias.detach())))
                seq[idx] = new_layer
            
            if isinstance(layer, TernaryConv2d):
                new_layer = nn.Conv2d(in_channels=layer.weight.shape[1], out_channels=layer.weight.shape[0], kernel_size=(layer.weight.shape[2], layer.weight.shape[3]), bias=True)
                new_layer.requires_grad_(False)
                new_layer.weight.copy_(torch.round(torch.tanh(layer.weight)))
                new_layer.bias.copy_(torch.round(torch.tanh(layer.bias)))
                seq[idx] = new_layer


class TernaryLayer(nn.Module):
    def __init__(self, module: nn.Module, functional: callable):
        super().__init__()
        self.weight = module.weight
        self.bias = module.bias
        self.f = functional

    def forward(self, x: torch.Tensor):
        weight = torch.tanh(self.weight)
        bias = torch.tanh(self.bias) if self.bias is not None else None
        return self.f(x, weight, bias)


class TernaryLinear(TernaryLayer):
    def __init__(self, **kwargs):
        super().__init__(nn.Linear(**kwargs), F.linear)


class TernaryConv2d(TernaryLayer):
    def __init__(self, **kwargs):
        super().__init__(nn.Conv2d(**kwargs), F.conv2d)
