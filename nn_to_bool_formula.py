from __future__ import annotations
from bool_formula import *
import torch.nn as nn
from typing import Optional
from enum import Enum
from collections import OrderedDict


class Act(Enum):
    SIGMOID = 1
    TANH = 2


class Neuron:
    def __init__(
        self,
        neurons_in: list[tuple[Neuron, float]] = [],
        bias: float = 0.0,
        activation_in: Act = Act.SIGMOID,
    ) -> None:
        self.neurons_in = neurons_in
        self.bias = bias
        self.activation_in = activation_in

    def __str__(self) -> str:
        rounded_weights = [round(t[1], 2) for t in self.neurons_in]
        ans = zip(rounded_weights, [t[0] for t in self.neurons_in])
        ans = " ".join(f"{t[0]}*{t[1]} " for t in ans)
        if self.bias:
            ans += str(round(self.bias, 2))
        return ans

    def __repr__(self) -> str:
        return str(self)

    def to_bool(self) -> Boolean:
        def to_bool_rec(
            neurons_in: list[tuple[str, float]],
            neuron_signs: list[bool],
            threshold: float,
            i: int = 0,
        ) -> Boolean:
            if threshold < 0:
                return Constant(True)
            if i == len(neurons_in):
                return Constant(False)

            name = neurons_in[i][0]
            weight = neurons_in[i][1]
            positive = not neuron_signs[i]

            # set to False
            term1 = to_bool_rec(neurons_in, neuron_signs, threshold, i + 1)
            term2 = AND(
                [
                    to_bool_rec(neurons_in, neuron_signs, threshold - weight, i + 1),
                    Literal(name, positive),
                ]
            )
            return OR([term1, term2])

        # sort neurons by their weight
        neurons_in = sorted(self.neurons_in, key=lambda x: abs(x[1]), reverse=True)

        # remember which weights are negative and then invert all of them (needed for negative numbers)
        negative = [tup[1] < 0 for tup in neurons_in]
        neurons_in = [(tup[0], abs(tup[1])) for tup in neurons_in]

        positive_weights = zip(negative, [tup[1] for tup in neurons_in])
        filtered_weights = filter(lambda tup: tup[0], positive_weights)
        bias_diff = sum(tup[1] for tup in filtered_weights)
        long_ans = to_bool_rec(neurons_in, negative, -self.bias + bias_diff)

        ans = simplified(long_ans)
        return ans


class NeuronNetwork:
    def __init__(self, net: nn.Module, varnames: Optional[list[str]] = None):
        self.neurons = OrderedDict()
        self.leaf = None
        self.varnames = varnames
        self.next_neuron_idx = 0

        if isinstance(net, nn.Sequential):
            first_layer = net[0]
            if not isinstance(first_layer, nn.Linear):
                raise ValueError("First layer must always be a linear layer.")
            shape_out, shape_in = first_layer.weight.shape
            if not varnames:
                varnames = [next(self.neuron_names()) for _ in range(shape_in)]
            if len(varnames) != shape_in:
                raise ValueError("varnames need same shape as input of first layer")
            # create a neuron for each of the input nodes in the first layer
            prev_layer = []
            for idx, name in enumerate(varnames):
                neuron = Neuron(neurons_in=[])
                self.neurons[name] = neuron
                prev_layer.append(neuron)

            for layer in net:
                if isinstance(layer, nn.Linear):
                    shape_out, shape_in = layer.weight.shape
                    weight = layer.weight.tolist()
                    bias = layer.bias.tolist()
                    new_layer = []
                    for idx in range(shape_out):
                        neurons_in = list(zip(prev_layer, weight[idx]))
                        neuron = Neuron(
                            neurons_in=neurons_in,
                            bias=bias[idx],
                        )
                        name = next(self.neuron_names())
                        self.neurons[name] = neuron
                        new_layer.append(neuron)
                    prev_layer = new_layer
            if len(prev_layer) == 1:
                self.leaf = prev_layer[0]

        else:
            raise ValueError("Only allows Sequential for now.")

    def __len__(self):
        return len(self.neurons)

    def __str__(self) -> str:
        return "\n".join(str(neuron) for neuron in self.neurons)

    def __getitem__(self, key: str):
        return self.neurons[key]

    def neuron_names(self):
        while True:
            self.next_neuron_idx += 1
            yield f"x_{self.next_neuron_idx}"

    def get_leaf_neuron(self) -> Neuron:
        return self.leaf


if __name__ == "__main__":
    x1 = Neuron()
    x2 = Neuron()
    x3 = Neuron()
    x4 = Neuron()
    b = Neuron("b", [(x1, 1.5), (x2, -1.4), (x3, 2.1), (x4, -0.3)], 1.0)
    print(b.to_bool())
