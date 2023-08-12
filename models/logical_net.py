import torch
import torch.nn as nn
from enum import Enum

class Activation(Enum):
    SIGMOID = 1
    TANH = 2


class LogicalNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.classifier = nn.Sequential()

    def forward(self, x):
        return self.classifier(x)

    def __str__(self) -> str:
        formulae = []
        for layer_idx, layer in enumerate(self.classifier):
            if not isinstance(layer, LogicalLayer): 
                raise ValueError("Only LogicalLayer modules allowed for string representation.")
            for neuron_idx, neuron in enumerate(layer.lin.weight):
                print(f"{layer_idx = } {neuron_idx = }")

        return "\n".join(formulae)
    


class LogicalLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=Activation.SIGMOID) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)
        self.activation = nn.Sigmoid() if activation == Activation.SIGMOID else nn.Tanh()
        

    def forward(self, x):
        return self.activation(self.lin(x))
    


