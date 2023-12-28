from collections.abc import Collection, Generator, Iterable, Mapping, Sequence
from copy import copy
from enum import Enum
from itertools import chain, combinations
from math import isclose
from typing import AbstractSet, Dict, Optional, Self, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from ckmeans_1d_dp import ckmeans
from urllib3 import encode_multipart_formdata

from bool_formula import AND, NOT, OR, Bool, Constant, Literal
from node import Key, Node, NodeGraph

sns.set()

Val = TypeVar("Val")


def powerset(it: Iterable[Val]) -> Iterable[Iterable[Val]]:
    s = list(it)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def possible_sums(vals: Collection[float]) -> Collection[float]:
    """
    Given n different float values, returns a list of length 2**n, consisting
    of each value that can be produced as a sum of a subset of values in vals.
    """
    return [sum(subset) for subset in powerset(vals)]


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


class QuantizedNeuron2:
    """Intermediate step between full-precision neurons and booleans as nodes."""

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
    """A full-precision neuron."""

    def __init__(
        self, name: Key, act: Activation, ins: Mapping[Key, float], bias: float
    ) -> None:
        super().__init__(name, ins.keys())
        self.act = act
        self.ins = ins
        self.bias = bias

    def sigmoid(self, x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            return np.exp(x) / (1.0 + np.exp(x))

    def __call__(self, vars: Mapping[Key, float]) -> float:
        ans = self.bias + sum(vars[n_key] * self.ins[n_key] for n_key in self.ins)
        return np.tanh(ans) if self.act == Activation.TANH else self.sigmoid(ans)


class QuantizedNeuron(Neuron[Key]):
    """A neuron that was quantized to a step function with either 0 or 1 steps."""

    def __init__(self, n: Neuron[Key], data_x: Collection[float]) -> None:
        super().__init__(n.key, n.act, n.ins, n.bias)

        data_x = possible_sums(data_x)
        data_x = np.fromiter(data_x, dtype=float)
        data_x.sort()
        data_x += self.bias

        self.neuron = n
        data_y = np.tanh(data_x)
        ans = ckmeans(data_y, (1, 2))
        print(f"{data_x = }")
        print(f"{data_y = }")
        cluster = ans.cluster
        n_cluster = len(np.unique(cluster))
        self.y_centers = np.array([ans.centers[i] for i in range(n_cluster)])
        self.x_centers = np.arctanh(self.y_centers)
        self.y_thrs = (self.y_centers[:-1] + self.y_centers[1:]) / 2
        self.x_thrs = np.arctanh(self.y_thrs)

    def __call__(self, vars: Mapping[Key, float]) -> float:
        ans = self.bias + sum(vars[n_key] * self.ins[n_key] for n_key in self.ins)
        for idx, x_thr in enumerate(self.x_thrs):
            if x_thr > ans:
                return self.y_centers[idx]
        return self.y_centers[-1]


class BooleanNeuron(Node[Key, bool]):
    def __init__(self, key: Key) -> None:
        super().__init__(
            key,
        )

    def __call__(self, vars: Mapping[Key, bool]) -> bool:
        return False


def to_neurons(net: nn.Sequential, vars: list[str]) -> list[Neuron[str]]:
    def name_generator():
        idx = 0
        while True:
            idx += 1
            yield f"h{idx}"

    name_gen = name_generator()
    ans: list[Neuron[str]] = []
    names = copy(vars)
    shape_out, shape_in = net[0].weight.shape
    assert (
        len(vars) == shape_in
    ), f"vars needs same shape as first layer input, but got: {len(vars)} != {shape_in}."
    n_layers = len(net)
    if isinstance(net[-1], nn.Flatten):
        n_layers -= 1  # ignore the last layer if it's a Flatten layer.
    for idx in range(0, n_layers, 2):
        lin_layer = net[idx]
        act_layer = net[idx + 1]
        assert isinstance(lin_layer, nn.Linear)
        assert isinstance(act_layer, nn.Sigmoid) or isinstance(act_layer, nn.Tanh)
        act = (
            Activation.SIGMOID if isinstance(act_layer, nn.Sigmoid) else Activation.TANH
        )
        shape_out, shape_in = lin_layer.weight.shape
        weight = lin_layer.weight.tolist()
        bias = lin_layer.bias.tolist()
        in_names = names[-shape_in:]
        for j in range(shape_out):
            new_name = next(name_gen)
            ins = {name: weight[j][i] for i, name in enumerate(in_names)}
            neuron = Neuron(new_name, act, ins, bias[j])
            ans.append(neuron)
            names.append(new_name)
    return ans


def to_vars(t: torch.Tensor, names: list[str]) -> Dict[str, float]:
    t_list = t.tolist()
    assert len(t_list) == len(names)
    return {names[i]: t_list[i] for i in range(len(t_list))}


def main():
    keys = ["a1", "a2", "a3"]
    in_neurons: dict[str, float] = {"a1": -1.55, "a2": -1.54, "a3": 1.6}
    bias = -0.86
    n1 = Neuron("n1", Activation.TANH, in_neurons, bias)

    q_n1 = QuantizedNeuron(n1, in_neurons.values())

    neuron_graph = NodeGraph([n1])
    q_neuron_graph = NodeGraph([q_n1])
    input_vals = {"a1": 1.0, "a2": 1.0, "a3": 0.0}
    pset = powerset(keys)
    diffs = []
    for subset in pset:
        input_vals = {key: 1.0 if key in subset else 0.0 for key in keys}
        diff = np.abs(neuron_graph(input_vals) - q_neuron_graph(input_vals))
        diffs.append(diff)
    sns.histplot(diffs, bins=12)
    plt.show()

    n = 5
    keys = [f"a{i + 1}" for i in range(n)]
    model = nn.Sequential(
        nn.Linear(n, n + 1, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(n + 1, n - 1, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(n - 1, 1, dtype=torch.float64),
        nn.Sigmoid(),
        nn.Flatten(0),
    ).train(False)
    neurons = to_neurons(model, keys)
    neuron_graph = NodeGraph(neurons)
    random_x = torch.rand(n, dtype=torch.float64)
    random_input_vars = to_vars(random_x, keys)
    print(f"{model(random_x).item() = }")
    print(f"{neuron_graph(random_input_vars) = }")
    assert isclose(
        model(random_x).float(), neuron_graph(random_input_vars), rel_tol=1e-6
    )


if __name__ == "__main__":
    main()
