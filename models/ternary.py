import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from utilities import get_all_weights


def R(weights: torch.Tensor, a: float):
    s = torch.tanh(weights) ** 2
    return torch.sum((a - s) * s)


class TernaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        module = nn.Linear(in_features, out_features, bias)
        self.weight = module.weight
        self.bias = module.bias
        self.f = F.linear

    def forward(self, x: torch.Tensor):
        assert isinstance(self.weight, torch.Tensor)
        weight = torch.tanh(self.weight)
        bias = torch.tanh(self.bias) if self.bias is not None else None
        return self.f(x, weight, bias)


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

    def quantized(self, prune: bool = False):
        return QuantizedModel(self, prune)


class QuantizedModel(nn.Module):
    def __init__(
        self, ternary_model: TernaryModule, do_prune: bool, quantize_tanh=False
    ):
        super().__init__()
        self.do_prune = do_prune
        self.quantize_tanh = quantize_tanh
        self.classifier = copy.deepcopy(ternary_model.classifier)
        self.device = next(self.classifier.parameters()).device
        self.tanh_and_round()

        if self.do_prune:
            self.prune()

    def tanh_and_round(self):
        for idx, layer in enumerate(self.classifier):
            layer.zero_grad()
            if isinstance(layer, TernaryLinear):
                with torch.no_grad():
                    b = layer.bias is not None
                    new_layer = nn.Linear(
                        in_features=layer.weight.shape[1],
                        out_features=layer.weight.shape[0],
                        bias=b,
                    )
                    new_layer.weight.copy_(
                        torch.round(torch.tanh(layer.weight.detach()))
                    )
                    if b:
                        new_layer.bias.copy_(
                            torch.round(torch.tanh(layer.bias.detach()))
                        )
                    self.classifier[idx] = new_layer
            elif isinstance(layer, nn.Tanh):
                self.classifier[idx] = Sign()
            elif isinstance(layer, nn.Sigmoid):
                self.classifier[idx] = Greater_one_half()

    def find_relevant_dims(self):
        def last_linear_layer_():
            ans = -1
            for idx, layer in enumerate(self.classifier):
                if isinstance(layer, nn.Linear):
                    ans = idx
            return ans

        relevant_dims = []
        last_lin = last_linear_layer_()

        for idx, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Linear):
                relevant_weight = layer.weight != 0
                relevant_bias = (
                    (layer.bias != 0)
                    if layer.bias is not None
                    else torch.zeros(layer.weight.shape[1], dtype=torch.bool)
                )
                relevant_col = torch.any(relevant_weight, axis=0)
                relevant_row = torch.any(relevant_weight, axis=1)
                relevant_row = torch.logical_or(relevant_row, relevant_bias)
                if (
                    idx == last_lin
                ):  # so as to not mess with output shape, the output cannot be shrunk
                    relevant_row = torch.ones_like(relevant_row, dtype=torch.bool)
                relevant_dims.append((relevant_col, relevant_row))

        ans = []
        num_linears = len(relevant_dims)

        ans.append(relevant_dims[0][0].nonzero().flatten())
        for i in range(num_linears - 1):
            all_relevant = (
                torch.logical_and(relevant_dims[i][1], relevant_dims[i + 1][0])
                .nonzero()
                .flatten()
            )
            ans.append(all_relevant)
        ans.append(relevant_dims[-1][-1].nonzero().flatten())
        return ans

    def prune(self):
        self.relevant_dims = self.find_relevant_dims()

        i = 0
        for idx, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Linear):
                in_features = len(self.relevant_dims[i])
                out_features = len(self.relevant_dims[i + 1])
                bias = layer.bias is not None and layer.bias.sum().item() != 0
                new_layer = nn.Linear(in_features, out_features, bias, self.device)
                new_weight = torch.index_select(
                    layer.weight, 1, index=self.relevant_dims[i]
                )
                new_weight = torch.index_select(
                    new_weight, 0, index=self.relevant_dims[i + 1]
                )
                assert new_layer.weight.shape == new_weight.shape
                new_layer.weight = nn.parameter.Parameter(new_weight)
                if bias:
                    new_bias = torch.index_select(
                        layer.bias, 0, index=self.relevant_dims[i + 1]
                    )
                    new_layer.bias = nn.parameter.Parameter(new_bias)
                self.classifier[idx] = new_layer
                i += 1

    def complexity(self):
        weights = get_all_weights(self)
        return weights.abs().sum().item()

    def forward(self, x):
        if self.do_prune:
            x = torch.index_select(x, 1, self.relevant_dims[0])
        ans = self.classifier(x).flatten()
        return ans, ans


class Sign(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sign(x)


class Greater_one_half(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.where(x > 0.5, 1.0, 0.0)
