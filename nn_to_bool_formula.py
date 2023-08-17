from __future__ import annotations
from bool_formula import *
import torch.nn as nn
from typing import Optional
from enum import Enum


class Act(Enum):
    SIGMOID = 1
    TANH = 2


class Neuron:
    def __init__(
        self,
        name="",
        neurons_in: list[tuple[Neuron, float]] = [],
        bias: float = 0.0,
        activation_in: Act = Act.SIGMOID,
    ) -> None:
        self.name = name
        self.neurons_in = neurons_in
        self.bias = bias
        self.activation_in = activation_in

    def __str__(self) -> str:
        left_term = self.name
        rounded_weights = [round(n[1], 2) for n in self.neurons_in]
        right_term = zip(rounded_weights, [t[0].name for t in self.neurons_in])
        right_term = " ".join(f"{t[0]}*{t[1]} " for t in right_term)
        if self.bias:
            right_term += str(round(self.bias, 2))
        return f"{left_term} := {right_term}"

    def to_bool(self) -> Boolean:
        def to_bool_rec(
            neurons_in: list[tuple[Neuron, float]],
            neuron_signs: list[bool],
            threshold: float,
            i: int = 0,
        ) -> Boolean:
            if threshold < 0:
                return Constant(True)
            if i == len(neurons_in):
                return Constant(False)

            name = neurons_in[i][0].name
            positive = not neuron_signs[i]
            weight = neurons_in[i][1]

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
        self.neurons = set()
        self.leaf = None

        if isinstance(net, nn.Sequential):
            first_layer = net[0]
            if not isinstance(first_layer, nn.Linear):
                raise ValueError("First layer must always be a linear layer.")
            shape_out, shape_in = first_layer.weight.shape
            if not varnames:
                varnames = [next(self.neuron_names()) for i in range(shape_in)]
            # create a neuron vor the the input nodes in the input layer
            last_layer = []
            for idx, name in enumerate(varnames):
                neuron = Neuron(name)
                self.neurons.add(neuron)
                last_layer.append(neuron)

            current_activation = None
            for layer in net:
                if isinstance(layer, nn.Sigmoid):
                    current_activation = Act.SIGMOID
                elif isinstance(layer, nn.Linear):
                    shape_out, shape_in = layer.weight.shape
                    weight = layer.weight.tolist()
                    bias = layer.bias.tolist()
                    new_layer = []
                    for idx in range(shape_out):
                        neurons_in = list(zip(last_layer, weight[idx]))
                        neuron = Neuron(
                            name=next(self.neuron_names()),
                            neurons_in=neurons_in,
                            bias=bias[idx],
                        )
                        self.neurons.add(neuron)
                        new_layer.append(neuron)
                    last_layer = new_layer
            if len(last_layer) == 1:
                self.leaf = last_layer[0]
        else:
            raise ValueError("Only allows Sequential for now.")

    def neuron_names(self):
        next_neuron_idx = 0
        while True:
            next_neuron_idx += 1
            yield f"x_{next_neuron_idx}"

    def get_leaf_neuron(self) -> Neuron:
        return self.leaf


if __name__ == "__main__":
    x1 = Neuron("x1")
    x2 = Neuron("x2")
    x3 = Neuron("x3")
    x4 = Neuron("x4")
    b = Neuron("b", [(x1, 1.5), (x2, -1.4), (x3, 2.1), (x4, -0.3)], 1.0)
    print(b.to_bool())
