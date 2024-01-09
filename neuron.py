from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from copy import copy, deepcopy
from enum import Enum
from itertools import chain, combinations
from typing import Dict, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
from ckmeans_1d_dp import ckmeans

from bool_formula import AND, NOT, OR, PARITY, Bool, Constant, Literal, possible_data
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


class Neuron(Node):
    """A full-precision neuron."""

    def __init__(
        self,
        key: str,
        act: Activation,
        ins: MutableMapping[str, float],
        bias: float,
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

    def __call__(self, vars: Mapping[str, np.ndarray]) -> np.ndarray:
        x = self.bias + sum(vars[key] * self.ins[key] for key in self.ins)
        return self.act_fn(x)

    def __str__(self):
        act_str = {Activation.SIGMOID: "sig", Activation.TANH: "tanh"}[self.act]
        params = [f"{self.ins[key]:.2f} * {key}" for key in self.ins]
        ans = f"{self.key} := {act_str}({str.join(' + ', params)} + {self.bias:.2f})"
        return ans


class NeuronGraph(Graph):
    def __init__(self, neurons: Sequence[Neuron]) -> None:
        super().__init__(neurons)
        self.neurons: dict[str, Neuron] = {n.key: n for n in neurons}

    @classmethod
    def from_nn(cls, net: nn.Sequential, varnames: Sequence[str]):
        def key_gen():
            idx = 0
            while True:
                idx += 1
                yield f"h{idx}"

        names = key_gen()
        ans: list[Neuron] = []
        varnames = list(copy(varnames))
        shape_out, shape_in = net[0].weight.shape  # type: ignore
        assert (
            len(varnames) == shape_in
        ), f"vars needs same shape as first layer input, but got: {len(varnames)} != {shape_in}."
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
            weight: list[list[float]] = lin_layer.weight.tolist()
            bias = lin_layer.bias.tolist()
            in_keys = varnames[-shape_in:]
            for j in range(shape_out):
                new_name = next(names)
                ins = {key: weight[j][i] for i, key in enumerate(in_keys)}
                neuron = Neuron(new_name, act, ins, bias[j])
                neuron = Neuron(new_name, act, ins, bias[j])
                ans.append(neuron)
                varnames.append(new_name)
                if idx == n_layers - 2:
                    assert shape_out == 1
                    neuron.key = "target"
        return NeuronGraph(ans)


class QuantizedNeuron(Node):
    """A neuron that was quantized to a step function with 1 step."""

    def __init__(
        self,
        key: str,
        ins: Mapping[str, float],
        bias: float,
        x_thr=0.0,
        y_centers: list[float] = [0.0, 1.0],
    ) -> None:
        self.key = key
        self.bias = bias
        self.ins = ins
        self.x_thr = x_thr
        self.y_centers = y_centers

    def __call__(self, vars: Mapping[str, float | np.ndarray]) -> float | np.ndarray:
        ans = self.bias + sum(vars[n_key] * self.ins[n_key] for n_key in self.ins)
        return np.where(ans >= self.x_thr, self.y_centers[1], self.y_centers[0])

    def __str__(self):
        params = [f"{self.ins[key]:.2f} * {key}" for key in self.ins]
        cond = str.join("\t+ ", params)
        cond += f"\t+ {self.bias:.2f} >= {self.x_thr:.2f}"
        ans = (
            f"{self.key} := "
            + f"[ {self.y_centers[1]:.2f} ] IF [ "
            + cond
            + f" ] ELSE [ {self.y_centers[0]:.2f} ]"
        )
        return ans


class QuantizedNeuronGraph(Graph):
    def __init__(self, q_neurons: list[QuantizedNeuron]) -> None:
        super().__init__(q_neurons)

    @classmethod
    def from_neuron_graph(cls, ng: NeuronGraph, data: MutableMapping[str, np.ndarray]):
        new_ng = deepcopy(ng)
        q_neuron_graph = QuantizedNeuronGraph([])
        reverse_ins = new_ng.reverse_ins()
        graph_ins = new_ng.graph_ins()
        # print(f"{reverse_ins = }")
        # print(f"{graph_ins = }")
        for key, neuron in new_ng.neurons.items():
            ins = graph_ins[key]
            if key not in new_ng.out_keys:  # non-output nodes
                outs = reverse_ins[neuron.key]
                data_y = neuron(data)
                assert isinstance(data_y, np.ndarray)
                data[key] = data_y

                ck_ans = ckmeans(data_y, (2))
                # print(f"{ck_ans.tot_withinss / len(data_y)}")
                # print(f"{ck_ans.totss / len(data_y)}")
                cluster = ck_ans.cluster
                n_cluster = len(np.unique(ck_ans.cluster))
                assert n_cluster == 2, NotImplementedError

                y_centers = ck_ans.centers

                # the threshold for the two groups lies in between the largest
                # of the small group and the smallest of the large group
                max_0 = np.max(data_y[cluster == 0])
                min_1 = np.min(data_y[cluster == 1])
                y_thr = (max_0 + min_1) / 2
                x_thr = neuron.inv_act(y_thr)
                x_thrs = neuron.inv_act(np.array([max_0, min_1]))

                # TODO: delete the node and add it's center value to the bias of the target nodes

                # adjust other weights such that the new q_neuron can have y_centers [-1.0, 1.0]
                a = y_centers[0]
                k = y_centers[1] - y_centers[0]

                for out_key in outs:
                    out_neuron = new_ng.nodes[out_key]
                    assert isinstance(out_neuron, Neuron)
                    w = out_neuron.ins[neuron.key]
                    out_neuron.bias += a * w  # update bias
                    out_neuron.ins[neuron.key] = w * k  # update weight

                q_neuron = QuantizedNeuron(neuron.key, neuron.ins, neuron.bias - x_thr)
                q_neuron_graph.add(q_neuron)
            else:  # output nodes
                q_neuron = QuantizedNeuron(neuron.key, neuron.ins, neuron.bias)
                q_neuron_graph.add(q_neuron)

        return q_neuron_graph


class BooleanNeuron(Node):
    def __init__(self, q_neuron: QuantizedNeuron) -> None:
        self.q_neuron = q_neuron
        self.key = q_neuron.key
        self.ins = q_neuron.ins
        self.b_val = self.to_bool()

    def __call__(self, vars: MutableMapping[str, np.ndarray]) -> np.ndarray:
        return self.b_val(vars)

    def to_bool(self) -> Bool:
        def init_dp(n_ins: list[float]) -> list[list[Tuple[float, float]]]:
            dp = []
            n_vars = len(n_ins)
            curr_sum = 0.0
            dp.append([])
            dp[-1].append((False, (float("-inf"), -curr_sum)))
            dp[-1].append((True, (0.0, float("inf"))))
            for k in range(n_vars):
                curr_sum += n_ins[k]
                dp.append([])
                dp[-1].append((False, (float("-inf"), -curr_sum)))
                dp[-1].append((True, (0.0, float("inf"))))
            return dp

        def to_bool_rec(
            k: int, threshold: float, dp
        ) -> Tuple[Bool, Tuple[float, float]]:
            # how much one could posible add by setting everyr variable to 1
            max_sum = sum(n[1] for n in self.n_ins[k:])
            # if already positive, return True
            if threshold >= 0.0:
                return (Constant(np.array(True)), (-threshold, float("inf")))
            # if you can't reach positive values, return False
            if max_sum + threshold <= 0.0:
                return (
                    Constant(np.array(False)),
                    (float("-inf"), -(max_sum + threshold)),
                )

            key = self.n_ins[k][0]
            weight = self.n_ins[k][1]
            positive = not self.signs[k]

            # set to False
            (term1, (min1, max1)) = to_bool_rec(k + 1, threshold, dp)
            (term2, (min2, max2)) = to_bool_rec(k + 1, threshold + weight, dp)
            term2 = AND(
                Literal(key) if positive else NOT(Literal(key)),
                term2,
            )
            ans = (OR(term1, term2).simplified(), (max(min1, min2), min(max1, max2)))
            # add to dp

            return ans

        assert self.q_neuron.y_centers == [0.0, 1.0]
        assert self.q_neuron.x_thr == 0.0
        bias = self.q_neuron.bias

        # sort neurons by their weight
        ins = list(self.ins.items())
        ins = sorted(ins, key=lambda x: abs(x[1]), reverse=True)

        # remember which weights are negative and then invert all of them (needed for negative numbers)
        self.signs = [tup[1] < 0 for tup in ins]
        self.n_ins = [(tup[0], abs(tup[1])) for tup in ins]

        positive_weights = list(zip(self.signs, [tup[1] for tup in self.n_ins]))
        filtered_weights = list(filter(lambda tup: tup[0], positive_weights))
        bias_diff = sum(tup[1] for tup in filtered_weights)

        dp = init_dp([n[1] for n in self.n_ins])
        for row in dp:
            print(row)
        (term, (min_thr, max_thr)) = to_bool_rec(0, bias - bias_diff, dp)
        print(f"{min_thr = }")
        print(f"{max_thr = }")
        return term

    @classmethod
    def from_q_neuron(cls, q_neuron: QuantizedNeuron):
        return BooleanNeuron(q_neuron)

    def __str__(self) -> str:
        return f"{self.key} := {str(self.b_val)}"


class BooleanGraph(Graph):
    def __init__(self, bools: Sequence[BooleanNeuron]) -> None:
        super().__init__(bools)

    @classmethod
    def from_q_neuron_graph(cls, q_ng: QuantizedNeuronGraph):
        return BooleanGraph(
            [
                BooleanNeuron(q_n)
                for key, q_n in q_ng.nodes.items()
                if isinstance(q_n, QuantizedNeuron)
            ]
        )


def nn_to_rule_set(
    model: nn.Sequential, data: MutableMapping[str, np.ndarray], vars: Sequence[str]
):
    # transform the trained neural network to a directed graph of full-precision neurons
    neuron_graph = NeuronGraph.from_nn(model, vars)
    # transform the graph to a new graph of perceptrons with quantized step functions
    q_neuron_graph = QuantizedNeuronGraph.from_neuron_graph(neuron_graph, data)
    # transform the quantized graph to a set of if-then rules
    bool_graph = BooleanGraph.from_q_neuron_graph(q_neuron_graph)
    return (neuron_graph, q_neuron_graph, bool_graph)


def to_vars(t: torch.Tensor, names: list[str]) -> Dict[str, float]:
    t_list = t.tolist()
    assert len(t_list) == len(names)
    return {names[i]: t_list[i] for i in range(len(t_list))}


def main():
    keys = ["x1", "x2", "x3"]
    k = 10

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
    data = possible_data(keys)
    q_neuron_graph = QuantizedNeuronGraph.from_neuron_graph(neuron_graph, data)
    # bool_graph = BooleanGraph.from_q_neuron_graph(q_neuron_graph)
    parity = PARITY(keys)

    n_correct, n_total = 0.0, 0.0
    ng_pred = neuron_graph(data)
    q_ng_pred = q_neuron_graph(data)
    print(f"{ng_pred = }")
    print(f"{q_ng_pred = }")

    print(f"n_nodes: {n_nodes} | fidelity: {n_correct / n_total}")
    # print(str(bool_graph))
    # TODO: fix fidelity: it has to be 1.0 in this example
    # _ = input("Continue?")


if __name__ == "__main__":
    main()
