from collections.abc import Collection, Iterable, Mapping, Sequence
from copy import copy
from enum import Enum
from itertools import chain, combinations
from typing import Dict, Self, TypeVar

import numpy as np
import torch
import torch.nn as nn
from ckmeans_1d_dp import ckmeans

from bool_formula import AND, NOT, OR, Bool, Constant, Literal
from node import Graph, Node

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


class Neuron(Node[float]):
    """A full-precision neuron."""

    def __init__(
        self, key: str, act: Activation, ins: Mapping[str, float], bias: float
    ) -> None:
        super().__init__(key, ins.keys())
        self.key = key
        self.act = act
        self.ins = ins
        self.bias = bias

    def sigmoid(self, x):
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

    def act_fn(self, x):
        return np.tanh(x) if self.act == Activation.TANH else self.sigmoid(x)

    def inv_act(self, x):
        return (
            np.arctanh(x) if self.act == Activation.TANH else np.log(x) - np.log(1 - x)
        )

    def __call__(self, vars: Mapping[str, float]) -> float:
        x = self.bias + sum(vars[n_key] * self.ins[n_key] for n_key in self.ins)
        return self.act_fn(np.array([x]))[0]

    def __str__(self):
        act_str = {Activation.SIGMOID: "sig", Activation.TANH: "tanh"}[self.act]
        params = [f"{self.ins[key]} * {key}" for key in self.ins]
        ans = f"{self.key} := {act_str}({str.join(' + ', params)} + {self.bias})"
        return ans


class NeuronGraph(Graph[float]):
    def __init__(self, neurons: Sequence[Neuron]) -> None:
        super().__init__(neurons)
        self.neurons = neurons

    @classmethod
    def from_nn(cls, net: nn.Sequential, vars: list[str]) -> list[Neuron]:
        def key_gen():
            idx = 0
            while True:
                idx += 1
                yield f"h{idx}"

        names = key_gen()
        ans: list[Neuron] = []
        vars = copy(vars)
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
                Activation.SIGMOID
                if isinstance(act_layer, nn.Sigmoid)
                else Activation.TANH
            )
            shape_out, shape_in = lin_layer.weight.shape
            weight = lin_layer.weight.tolist()
            bias = lin_layer.bias.tolist()
            in_names = vars[-shape_in:]
            for j in range(shape_out):
                new_name = next(names)
                ins = {name: weight[j][i] for i, name in enumerate(in_names)}
                neuron = Neuron(new_name, act, ins, bias[j])
                ans.append(neuron)
                vars.append(new_name)
        return ans


class QuantizedNeuron(Node[float]):
    """A neuron that was quantized to a step function with either 0 or 1 steps."""

    def __init__(
        self, n: Neuron, x_thrs: Collection[float], y_centers: Sequence[float]
    ) -> None:
        super().__init__(n.key, n.ins)
        self.n = n
        self.x_thrs = x_thrs
        self.y_centers = y_centers

    def __call__(self, vars: Mapping[str, float]) -> float:
        ans = self.n.bias + sum(vars[n_key] * self.n.ins[n_key] for n_key in self.ins)
        for idx, x_thr in enumerate(self.x_thrs):
            if x_thr > ans:
                return self.y_centers[idx]
        return self.y_centers[-1]

    @classmethod
    def from_neuron(cls, n: Neuron, data_x: Collection[float] | None = None):
        if data_x is None:
            data_x = n.ins.values()
        data_x = possible_sums(data_x)
        data_x = np.fromiter(data_x, dtype=float)
        data_x.sort()
        data_x += n.bias
        data_y = n.act_fn(data_x)

        if len(data_y) == 1:
            y_centers = data_y
        else:
            ans = ckmeans(data_y, 2)
            cluster = ans.cluster
            n_cluster = len(np.unique(cluster))
            y_centers = np.array([ans.centers[i] for i in range(n_cluster)])

        y_thrs = (y_centers[:-1] + y_centers[1:]) / 2
        x_thrs = n.inv_act(y_thrs)
        return QuantizedNeuron(n, x_thrs, list(y_centers))


class QuantizedNeuronGraph(Graph[float]):
    def __init__(self, q_neurons: Sequence[QuantizedNeuron]) -> None:
        super().__init__(q_neurons)
        self.q_neurons = q_neurons

    @classmethod
    def from_neuron_graph(cls, ng: NeuronGraph):
        q_neurons = [QuantizedNeuron.from_neuron(n) for n in ng.neurons]
        return QuantizedNeuronGraph(q_neurons)


class BooleanNeuron(Node[bool]):
    def __init__(self, q_neuron: QuantizedNeuron) -> None:
        super().__init__(q_neuron.key, q_neuron.ins)
        self.q_n = q_neuron
        self.b_val = self.to_bool()

    def __call__(self, vars: Mapping[str, bool]) -> bool:
        return self.b_val(vars)

    def to_bool(self) -> Bool:
        def to_bool_rec(
            neurons_in: list[tuple[Self, float]],
            neuron_signs: list[bool],
            threshold: float,
            i: int = 0,
        ) -> Bool:
            if threshold >= 0:
                return Constant(True)
            if threshold < -sum(n[1] for n in neurons_in[i:]):
                return Constant(False)

            key = neurons_in[i][0].key
            weight = neurons_in[i][1]
            positive = not neuron_signs[i]

            # set to False
            term1 = to_bool_rec(neurons_in, neuron_signs, threshold, i + 1)

            term2 = to_bool_rec(neurons_in, neuron_signs, threshold + weight, i + 1)
            term2 = AND(
                Literal(key) if positive else NOT(Literal(key)),
                term2,
            )
            return OR(term1, term2).simplified()

        # step 1: adjust Neuron so that all activations from input are boolean, while preserving equality
        for idx, key in enumerate(self.q_n.n.ins):
            weight = self.q_n.n.ins[key]

            # a = self.q_n.y_centers[0]
            assert len(self.q_n.y_centers) == 2
            self.q_n.n.bias += weight * self.q_n.y_centers[0]

            # k = self.q_n.y_centers[1] - self.q_n.y_centers[0]
            k = self.q_n.y_centers[1] - self.q_n.y_centers[0]
            self.q_n.ins[key] *= 
            # TODO: this right here is not yet working!

        # sort neurons by their weight
        neurons_in = sorted(self.neurons_in, key=lambda x: abs(x[1]), reverse=True)

        # remember which weights are negative and then invert all of them (needed for negative numbers)
        negative = [tup[1] < 0 for tup in neurons_in]
        neurons_in = [(tup[0], abs(tup[1])) for tup in neurons_in]

        positive_weights = list(zip(negative, [tup[1] for tup in neurons_in]))
        filtered_weights = list(filter(lambda tup: tup[0], positive_weights))
        bias_diff = sum(tup[1] for tup in filtered_weights)

        return to_bool_rec(neurons_in, negative, self.bias - bias_diff)

    @classmethod
    def from_q_neuron(cls, q_neuron: QuantizedNeuron):
        return BooleanNeuron(q_neuron)


class BooleanGraph(Graph[bool]):
    def __init__(self, bools: Sequence[BooleanNeuron]) -> None:
        super().__init__(bools)

    @classmethod
    def from_q_neuron_graph(cls, q_ng: QuantizedNeuronGraph):
        return BooleanGraph([BooleanNeuron(q_neuron) for q_neuron in q_ng.q_neurons])


def to_vars(t: torch.Tensor, names: list[str]) -> Dict[str, float]:
    t_list = t.tolist()
    assert len(t_list) == len(names)
    return {names[i]: t_list[i] for i in range(len(t_list))}


def main():
    keys = ["x1", "x2", "x3"]
    k = 10

    h1 = Neuron("h1", Activation.SIGMOID, {"x1": k, "x2": k}, -0.5 * k)
    h2 = Neuron("h2", Activation.SIGMOID, {"x1": -k, "x2": -k}, 0.5 * k)
    h3 = Neuron("h3", Activation.SIGMOID, {"x3": k}, -0.5 * k)
    h4 = Neuron("h4", Activation.SIGMOID, {"h1": k, "h2": k}, -1.5 * k)
    h5 = Neuron("h5", Activation.SIGMOID, {}, -100)
    h6 = Neuron("h6", Activation.SIGMOID, {"h3": k}, -0.5 * k)
    h7 = Neuron("h7", Activation.SIGMOID, {"h4": k, "h6": k}, -0.5 * k)
    h8 = Neuron("h8", Activation.SIGMOID, {}, -100)
    h9 = Neuron("h9", Activation.SIGMOID, {"h4": -k, "h6": -k}, 0.5 * k)
    h10 = Neuron("h10", Activation.SIGMOID, {"h7": k, "h9": k}, -0.5 * k)

    neurons = [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10]

    for n_nodes in range(1, 11):
        # bool_neurons = [BooleanNeuron(q_neuron) for q_neuron in q_neurons]
        neuron_graph = NeuronGraph(neurons[:n_nodes])
        q_neuron_graph = QuantizedNeuronGraph.from_neuron_graph(neuron_graph)

        bool_graph = BooleanGraph.from_q_neuron_graph(q_neuron_graph)

        n_correct, n_total = 0.0, 0.0
        for subset in powerset(keys):
            vars = {key: 1.0 if key in subset else 0.0 for key in keys}
            bool_vars = {key: True if key in subset else False for key in keys}
            neuron_output = neuron_graph(vars, neurons[n_nodes - 1].key)
            q_neuron_output = q_neuron_graph(vars, neurons[n_nodes - 1].key)
            bool_output = bool_graph(bool_vars, neurons[n_nodes - 1].key)

            # print(f"{vars = }")
            # print(f"{neuron_output}")
            # print(f"{q_neuron_output}")
            # print()
            if round(neuron_output) == round(q_neuron_output):
                n_correct += 1.0
            n_total += 1.0
        print(f"n_nodes: {n_nodes} | fidelity: {n_correct / n_total}")
        # print(f"{bool_neuron_graph(bool_vars)}")


if __name__ == "__main__":
    main()
