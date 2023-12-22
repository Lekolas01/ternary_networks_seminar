from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Optional, Self

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ckmeans_1d_dp import ckmeans

from bool_formula import AND, NOT, OR, Bool, Constant, Literal
from node import Key, Node, NodeGraph, Val


class Activation(Enum):
    SIGMOID = 1
    TANH = 2


class Neuron2:
    def __init__(
        self,
        name: str,
        neurons_in: list[tuple[Self, float]] = [],
        bias: float = 0.0,
        activation_in: Activation = Activation.SIGMOID,
    ) -> None:
        self.name = name
        self.neurons_in = neurons_in
        self.bias = bias
        self.activation = activation_in

    def __str__(self) -> str:
        right_term = ""
        act_str = "sig" if self.activation == Activation.SIGMOID else "tanh"

        if len(self.neurons_in) >= 1:
            neuron, weight = self.neurons_in[0]
            right_term += f"{round(weight, 2):>5}*{neuron.name:<5}"
        for neuron, weight in self.neurons_in[1:]:
            weight_sign = "+" if weight >= 0 else "-"
            right_term += f"{weight_sign}{round(abs(weight), 2):>5}*{neuron.name:<5}"

        if self.bias:
            weight_sign = "+" if self.bias >= 0 else "-"
            right_term += f"{weight_sign}{round(abs(self.bias), 2)}"
        if not right_term:
            right_term = self.name
        return f"{self.name:>8}  :=  {act_str}({right_term})"

    def __repr__(self) -> str:
        return f'Neuron("{str(self)}")'

    def to_bool(self) -> Bool:
        def to_bool_rec(
            neurons_in: list[tuple[Self, float]],
            neuron_signs: list[bool],
            threshold: float,
            i: int = 0,
            # dp: list[list[tuple[float, float, Bool]]] | None = None,
        ) -> Bool:
            def find_in_ranges(
                ranges: list[tuple[float, float, Bool]], val: float
            ) -> tuple[int, bool]:
                left, right = 0, len(ranges)
                while left < right:
                    mid = (left + right) // 2
                    curr_range_left, curr_range_right, _ = ranges[mid]
                    if curr_range_left <= val <= curr_range_right:
                        return mid, True
                    elif curr_range_right < val:
                        left = mid + 1
                    else:
                        right = mid
                return right, False

            # if dp is None:
            #    dp = [[] for j in range(len(neurons_in))]

            if threshold >= 0:
                return Constant(True)
            if threshold < -sum(n[1] for n in neurons_in[i:]):
                return Constant(False)
            # idx, is_found = find_in_ranges(dp[i], threshold)
            #    return dp[i][idx]

            name = neurons_in[i][0].name
            weight = neurons_in[i][1]
            positive = not neuron_signs[i]

            # set to False
            term1 = to_bool_rec(neurons_in, neuron_signs, threshold, i + 1)

            term2 = to_bool_rec(neurons_in, neuron_signs, threshold + weight, i + 1)
            term2 = AND(
                Literal(name) if positive else NOT(Literal(name)),
                term2,
            )
            # TODO: add to dp
            return OR(term1, term2).simplified()

        # step 1: adjust Neuron so that all activations from input are boolean, while preserving equality
        for idx, (neuron_in, weight) in enumerate(self.neurons_in):
            if (
                not isinstance(neuron_in, InputNeuron)
                and neuron_in.activation == Activation.TANH
            ):
                # a = -1
                self.bias -= weight

                # k = 2
                temp = self.neurons_in[idx]
                temp = list(temp)
                temp[1] *= 2  # type: ignore
                self.neurons_in[idx] = tuple(temp)  # type: ignore

        # sort neurons by their weight
        neurons_in = sorted(self.neurons_in, key=lambda x: abs(x[1]), reverse=True)

        # remember which weights are negative and then invert all of them (needed for negative numbers)
        negative = [tup[1] < 0 for tup in neurons_in]
        neurons_in = [(tup[0], abs(tup[1])) for tup in neurons_in]

        positive_weights = list(zip(negative, [tup[1] for tup in neurons_in]))
        filtered_weights = list(filter(lambda tup: tup[0], positive_weights))
        bias_diff = sum(tup[1] for tup in filtered_weights)

        return to_bool_rec(neurons_in, negative, self.bias - bias_diff)


class QuantizedNeuron:
    """Intermediate step between full-precision neurons and booleans as nodes. it contains both you should wrap this around, this sould not stay like this..."""

    def __init__(self, neuron: Neuron2, x: np.ndarray) -> None:
        self.neuron = neuron
        self.input_dist = x
        y = np.tanh(x)
        ans = ckmeans(np.tanh(x), k=(1, 2))
        assert all(
            ans.cluster[i] <= ans.cluster[i + 1] for i in range(len(ans.cluster) - 1)
        ), "clusters must be sorted by their mean."
        y_means = np.array([c for c in ans.centers if c != 0])

        y_thrs = (y_means[:-1] + y_means[1:]) / 2
        x_thrs = np.arctanh(y_thrs)


class InputNeuron(Neuron2):
    def __init__(self, name: str) -> None:
        self.name = name

    def to_bool(self) -> Bool:
        return Literal(self.name)

    def __str__(self) -> str:
        return f'InputNeuron("{self.name}")'


class NeuronGraph2:
    def __init__(self, vars: Optional[list[str]], net: Optional[nn.Sequential] = None):
        self.new_neuron_idx = 1  # for naming new neurons
        self.neurons: list[Neuron2] = []  # collection of all neurons added to Network
        self.neuron_names: set[str] = set()  # keeps track of the names of all neurons
        if net:
            assert vars is not None
            self.add_module(net, vars)

    def __len__(self):
        return len(self.neurons)

    def __str__(self) -> str:
        return "\n".join(str(neuron) for neuron in self.neurons)

    def add_module(self, net: nn.Sequential, input_vars: list[str]):
        self.input_vars = input_vars  # the names of the input variables
        first_layer = net[0]
        if not isinstance(first_layer, nn.Linear):
            raise ValueError("First layer must always be a linear layer.")
        shape_out, shape_in = first_layer.weight.shape
        if len(self.input_vars) != shape_in:
            raise ValueError("varnames need same shape as input of first layer")

        # create a neuron for each of the input nodes in the first layer
        for idx, name in enumerate(self.input_vars):
            new_input_neuron = InputNeuron(name)
            self.add(new_input_neuron)

        ll_start, ll_end = 0, len(self.neurons)
        curr_act = Activation.SIGMOID
        for idx, layer in enumerate(net):
            if isinstance(layer, nn.Linear):
                next_layer = net[idx + 1]
                if isinstance(next_layer, nn.Sigmoid):
                    curr_act = Activation.SIGMOID
                elif isinstance(next_layer, nn.Tanh):
                    curr_act = Activation.TANH

                shape_out, shape_in = layer.weight.shape
                weight = layer.weight.tolist()
                bias = layer.bias.tolist()

                for idx in range(shape_out):
                    neurons_in = list(zip(self.neurons[ll_start:ll_end], weight[idx]))
                    name = self._new_name()
                    neuron = Neuron2(
                        name,
                        neurons_in=neurons_in,
                        bias=bias[idx],
                        activation_in=curr_act,
                    )
                    self.add(neuron)
                ll_start, ll_end = ll_end, len(self.neurons)

        # rename the last variable, so it is distinguishable from the rest
        self.rename(self.target(), "target")

    def add(self, neuron: Neuron2):
        assert neuron.name not in self.neuron_names
        self.neurons.append(neuron)
        self.neuron_names.add(neuron.name)

    def rename(self, neuron: Neuron2, new_name: str):
        assert neuron in self.neurons
        assert new_name not in self.neuron_names
        neuron.name = new_name

    def _new_name(self):
        while f"h{self.new_neuron_idx}" in self.neuron_names:
            self.new_neuron_idx += 1
        return f"h{self.new_neuron_idx}"

    def target(self) -> Neuron2:
        return self.neurons[-1]


class Neuron(Node[Key, float]):
    """Full-precision neuron."""

    def __init__(
        self, name: Key, act: Activation, ins: Mapping[Key, float], bias: float
    ) -> None:
        super().__init__(name, ins.keys())
        self.act = act
        self.in_neurons = ins
        self.bias = bias

    def __call__(self, var_setting: Mapping[Key, float]) -> float:
        ans = self.bias + sum(
            var_setting[n_key] * self.in_neurons[n_key] for n_key in self.in_neurons
        )
        return np.tanh(ans) if self.act == Activation.TANH else 1 / (1 + np.exp(-ans))


class NeuronGraph(NodeGraph[Key, float]):
    def __init__(self, nodes: Sequence[Node[Key, float]]) -> None:
        super().__init__(nodes)


def main():
    n1 = Neuron("n1", Activation.TANH, {"a1": 1.2, "a2": 2.4}, 3.5)
    n2 = Neuron("n2", Activation.TANH, {"n1": 2.0}, 0.0)
    neuron_graph = NeuronGraph([n1, n2])
    print(neuron_graph({"a1": 0, "a2": 1.0}))


if __name__ == "__main__":
    main()
