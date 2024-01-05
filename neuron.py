from collections.abc import Collection, Iterable, Mapping, MutableMapping, Sequence
from copy import copy
from enum import Enum
from itertools import chain, combinations
from typing import Dict, Self, TypeVar

import numpy as np
import torch
import torch.nn as nn
from ckmeans_1d_dp import ckmeans

from bool_formula import AND, NOT, OR, PARITY, Bool, Constant, Literal
from node import Graph, Node

Val = TypeVar("Val")


def powerset(it: Iterable[Val]) -> Iterable[Iterable[Val]]:
    s = list(it)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def possible_sums(vals: Iterable[float]) -> list[float]:
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
        self, key: str, act: Activation, ins: MutableMapping[str, float], bias: float
    ) -> None:
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
        params = [f"{self.ins[key]:.2f} * {key}" for key in self.ins]
        ans = f"{self.key} := {act_str}({str.join(' + ', params)} + {self.bias:.2f})"
        return ans


class NeuronGraph(Graph[float]):
    def __init__(self, neurons: Sequence[Neuron]) -> None:
        super().__init__(neurons)
        self.neurons = neurons

    @classmethod
    def from_nn(cls, net: nn.Sequential, vars: list[str]):
        def key_gen():
            idx = 0
            while True:
                idx += 1
                yield f"h{idx}"

        names = key_gen()
        ans: list[Neuron] = []
        vars = copy(vars)
        shape_out, shape_in = net[0].weight.shape  # type: ignore
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
                if idx == n_layers - 2:
                    assert shape_out == 1
                    neuron.key = "target"
        return NeuronGraph(ans)


class QuantizedNeuron(Node[float]):
    """A neuron that was quantized to a step function with either 0 or 1 steps."""

    def __init__(
        self, n: Neuron, x_thrs: Sequence[float], y_centers: Sequence[float]
    ) -> None:
        self.key = n.key
        self.ins = n.ins
        self.n = n
        self.x_thrs = x_thrs
        self.y_centers = y_centers

    def __call__(self, vars: Mapping[str, float]) -> float:
        ans = self.n.bias + sum(vars[n_key] * self.ins[n_key] for n_key in self.ins)
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
            y_thrs = y_centers[0]
        else:
            ans = ckmeans(data_y, 2)
            n_cluster = len(np.unique(ans.cluster))
            y_centers = np.array([ans.centers[i] for i in range(n_cluster)])
            # find first value that corresponds to second group
            new_grp_idx = np.nonzero(ans.cluster == 1)[0][0]
            y_thrs = (data_y[new_grp_idx - 1] + data_y[new_grp_idx]) / 2

        x_thrs = n.inv_act(y_thrs)
        return QuantizedNeuron(n, x_thrs, list(y_centers))

    def __str__(self):
        params = [f"{self.ins[key]:.2f} * {key}" for key in self.ins]
        cond = str.join("\t+ ", params)
        cond += f"\t+ {self.n.bias:.2f} >= {self.x_thrs:.2f}"
        ans = (
            f"{self.key} := "
            + f"[ {self.y_centers[1]:.2f} ] IF [ "
            + cond
            + f" ] ELSE [ {self.y_centers[0]:.2f} ]"
        )
        return ans


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
        self.q_neuron = q_neuron
        self.key = q_neuron.key
        self.ins = q_neuron.ins
        self.b_val = self.to_bool()

    def __call__(self, vars: MutableMapping[str, bool]) -> bool:
        return self.b_val(vars)

    def to_bool(self) -> Bool:
        def to_bool_rec(
            neurons_in: list[tuple[str, float]],
            neuron_signs: list[bool],
            threshold: float,
            i: int = 0,
        ) -> Bool:
            if threshold >= 0:
                return Constant(True)
            if threshold < -sum(n[1] for n in neurons_in[i:]):
                return Constant(False)

            key = neurons_in[i][0]
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

        x_thrs = self.q_neuron.x_thrs
        y_centers = self.q_neuron.y_centers
        bias = self.q_neuron.n.bias
        # bias += x_thrs[0]

        # step 1: adjust Neuron so that all activations from input are boolean, while preserving equality
        for idx, key in enumerate(self.ins):
            weight = self.ins[key]

            # a = self.q_n.y_centers[0]
            assert len(y_centers) == 2
            bias += weight * y_centers[0]

            # k = self.q_n.y_centers[1] - self.q_n.y_centers[0]
            k = y_centers[1] - y_centers[0]
            self.ins[key] = self.ins[key] * k

        # sort neurons by their weight
        ins = list(self.ins.items())

        ins = sorted(ins, key=lambda x: abs(x[1]), reverse=True)

        # remember which weights are negative and then invert all of them (needed for negative numbers)
        negative = [tup[1] < 0 for tup in ins]
        ins = [(tup[0], abs(tup[1])) for tup in ins]

        positive_weights = list(zip(negative, [tup[1] for tup in ins]))
        filtered_weights = list(filter(lambda tup: tup[0], positive_weights))
        bias_diff = sum(tup[1] for tup in filtered_weights)

        return to_bool_rec(ins, negative, bias - bias_diff)

    @classmethod
    def from_q_neuron(cls, q_neuron: QuantizedNeuron):
        return BooleanNeuron(q_neuron)

    def __str__(self) -> str:
        return f"{self.key} := {str(self.b_val)}"


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
    k = 6

    h1 = Neuron("h1", Activation.SIGMOID, {"x1": k, "x2": k}, -0.5 * k)
    h2 = Neuron("h2", Activation.SIGMOID, {"x1": -k, "x2": -k}, 1.5 * k)
    h3 = Neuron("h3", Activation.SIGMOID, {"x3": k}, -0.5 * k)
    h4 = Neuron("h4", Activation.SIGMOID, {"h1": k, "h2": k}, -1.5 * k)
    h5 = Neuron("h5", Activation.SIGMOID, {}, -100)
    h6 = Neuron("h6", Activation.SIGMOID, {"h3": k}, -0.5 * k)
    h7 = Neuron("h7", Activation.SIGMOID, {"h4": -k, "h6": k}, -0.5 * k)
    h8 = Neuron("h8", Activation.SIGMOID, {}, -100)
    h9 = Neuron("h9", Activation.SIGMOID, {"h4": k, "h6": -k}, -0.5 * k)
    h10 = Neuron("target", Activation.SIGMOID, {"h7": k, "h9": k}, -0.5 * k)

    neurons = [h1, h2, h3, h4, h5, h6, h7, h8, h9, h10]

    n_nodes = 10
    neuron_graph = NeuronGraph(neurons[:n_nodes])
    q_neuron_graph = QuantizedNeuronGraph.from_neuron_graph(neuron_graph)
    bool_graph = BooleanGraph.from_q_neuron_graph(q_neuron_graph)
    parity = PARITY(keys)

    n_correct, n_total = 0.0, 0.0
    for subset in powerset(keys):
        vars = {key: 1.0 if key in subset else 0.0 for key in keys}
        bool_vars = {key: True if key in subset else False for key in keys}
        neuron_output = neuron_graph(vars)
        # q_neuron_output = q_neuron_graph(vars, neurons[n_nodes - 1].key)
        # bool_output = bool_graph(bool_vars, neurons[n_nodes - 1].key)
        neuron_b = True if neuron_output >= 0.5 else False
        # q_neuron_b = True if q_neuron_output >= 0.5 else False
        p_out = parity(bool_vars)
        # print(f"{vars = }")
        # print(f"{neuron_output}")
        # print(f"{q_neuron_output}")
        # print(f"{bool_output}")
        # print()

        if neuron_b == p_out:
            n_correct += 1.0
        # else:
        #    print(f"{subset}")
        #    print(f"{neuron_b} {q_neuron_b} {bool_output} {p_out}")
        #    print(f"{neuron_output} {q_neuron_output}")
        #    print()

        n_total += 1.0
    print(f"n_nodes: {n_nodes} | fidelity: {n_correct / n_total}")
    print(str(bool_graph))
    # TODO: fix fidelity: it has to be 1.0 in this example
    # _ = input("Continue?")


if __name__ == "__main__":
    main()
