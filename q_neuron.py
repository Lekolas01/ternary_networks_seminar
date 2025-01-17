import copy
import enum
from collections.abc import Mapping, MutableMapping
from typing import Self, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ckmeans_1d_dp import ckmeans

from bool_formula import PARITY, Activation, overlap, possible_data
from neuron import Neuron, NeuronGraph
from node import Graph, Node


class Perceptron(Node):
    """A neuron with two-interval step function as activation."""

    def __init__(
        self,
        key: str,
        ins: MutableMapping[str, float],
        bias: float,
        y_centers: list[float] = [0.0, 1.0],
    ) -> None:
        self.key = key
        self.bias = bias
        self.ins = ins
        self.y_centers = y_centers

    def __call__(self, vars: Mapping[str, float | np.ndarray]) -> float | np.ndarray:
        ans = self.bias + sum(vars[n_key] * self.ins[n_key] for n_key in self.ins)
        return np.where(ans >= 0, self.y_centers[1], self.y_centers[0])

    def __str__(self):
        temp = list(self.ins.items())
        temp.sort(key=lambda x: -abs(x[1]))
        temp = {k: v for k, v in temp}
        params = [f"{self.ins[key]:.3f} * {key}" for key in temp]
        cond = str.join("\t+ ", params)
        cond += f"\t+ {self.bias:.3f} >= 0"
        ans = (
            f"{self.key} := "
            + f"[ {self.y_centers[1]:.3f} ] IF [ "
            + cond
            + f" ] ELSE [ {self.y_centers[0]:.3f} ]"
        )
        return ans

    def prune(self, thr: float):
        # prune inputs with less impact than abs(thr)
        self.ins = {k: v for k, v in self.ins.items() if abs(v) >= thr}


def from_neuron(
    neuron: Neuron, data: MutableMapping[str, np.ndarray]
) -> tuple[np.ndarray, Perceptron]:
    y = neuron(data)  # calculate distribution of node output on training data
    assert isinstance(y, np.ndarray), "y is not a np array"

    ck_ans = ckmeans(y, (2))  # cluster the distribution to two clusters
    cluster = ck_ans.cluster
    n_cluster = len(np.unique(ck_ans.cluster))

    assert n_cluster == 2, "Not implemented"

    y_centers = list(ck_ans.centers[:2])

    # the threshold for the two groups lies in between the largest
    # of the small group and the smallest of the large group
    max_0 = np.max(y[cluster == 0])
    min_1 = np.min(y[cluster == 1])
    y_thr = (max_0 + min_1) / 2
    x_thr = neuron.inv_act(y_thr)

    q_neuron = Perceptron(neuron.key, neuron.ins, neuron.bias - x_thr, y_centers)
    return (y, q_neuron)


class QuantizedNeuronGraph(Graph):
    def __init__(self, q_neurons: list[Perceptron]) -> None:
        super().__init__(q_neurons)

    @classmethod
    def from_neuron_graph(cls, ng: NeuronGraph, data: MutableMapping[str, np.ndarray]):
        q_neuron_graph = QuantizedNeuronGraph([])
        out_key = next(iter(ng.out_keys))
        for key, neuron in ng.neurons.items():
            data[key], q_neuron = from_neuron(neuron, data)
            if q_neuron.key == out_key:
                q_neuron.y_centers = [0.0, 1.0]
            q_neuron_graph.add(q_neuron)
        return q_neuron_graph

    def __repr__(self):
        return str(self)

    def __call__(self, data: pd.DataFrame | dict[str, np.ndarray]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            keys = list(data.columns)
            data = {key: np.array(data[key], dtype=bool) for key in keys}
        return super().__call__(data)


class CustomLayer(nn.Module):
    def __init__(self, lin: nn.Linear, quantized: list[bool]):
        self.lin = lin
        self.quantized = quantized

    def forward(self, x):
        ans = self.lin(x)


class SparseLayer(nn.Module):
    def __init__(self, lin: nn.Linear):
        super(SparseLayer, self).__init__()
        self.lin = lin
        self.m = torch.empty(self.lin.weight.shape)

    def forward(self, x):
        w = self.lin * self.m
        ans = w(x)


class QuantizedLayer(nn.Module):
    def __init__(
        self,
        lin: nn.Linear,
        y_low: torch.Tensor,
        y_high: torch.Tensor,
    ):
        super(QuantizedLayer, self).__init__()
        self.lin = lin
        self.y_low = y_low
        self.y_high = y_high

    def forward(self, x: torch.Tensor):
        ans = self.lin(x)
        return torch.where(ans >= 0, self.y_high, self.y_low)

    def __str__(self):
        return f"QuantizedLayer(in_features={self.lin.in_features}, out_features={self.lin.out_features})"

    def __repr__(self):
        return str(self)


def QNG_from_QNN(model: nn.Sequential, varnames: Sequence[str]) -> QuantizedNeuronGraph:
    def key_gen():
        idx = 0
        while True:
            idx += 1
            yield f"h{idx}"

    if isinstance(model[-1], nn.Flatten):
        model = model[:-1]
    assert all([isinstance(l, QuantizedLayer) for l in model])
    names = key_gen()
    ans: list[Perceptron] = []
    varnames = list(copy.copy(varnames))
    curr_shape = len(varnames)
    for idx, q_layer in enumerate(model):
        out_features, in_features = q_layer.lin.weight.shape
        y_low, y_high = q_layer.y_low.numpy(), q_layer.y_high.numpy()
        if y_low.ndim == 0:
            y_low, y_high = [y_low.item()], [y_high.item()]
        in_keys = varnames[-curr_shape:]
        assert curr_shape == in_features
        curr_shape = out_features
        weight: list[list[float]] = q_layer.lin.weight.tolist()
        bias = q_layer.lin.bias.tolist()
        for j in range(curr_shape):
            new_name = next(names)
            try:
                ins = {key: weight[j][i] for i, key in enumerate(in_keys)}
            except:
                temp = 0
            q_neuron = Perceptron(new_name, ins, bias[j], [y_low[j], y_high[j]])
            ans.append(q_neuron)
            varnames.append(new_name)
            if idx == len(model) - 1:
                assert curr_shape == 1
                q_neuron.key = "target"
    return QuantizedNeuronGraph(ans)


def normalized(q_ng: QuantizedNeuronGraph) -> QuantizedNeuronGraph:
    """Normalize the nodes in the Quantized Neuron graph.
    This does not change the output of the graph."""
    ans = QuantizedNeuronGraph([])
    order = q_ng.topological_order()
    for key in order:
        neuron = q_ng[key]
        print(neuron)
        q_neuron = copy.copy(neuron)  # TODO
        ans.add(q_neuron)
    return ans


def main():
    keys = ["x1", "x2", "x3"]
    k = 10

    h1 = Neuron("h1", Activation.SIGMOID, {"x1": k, "x2": k}, -0.5 * k)
    h2 = Neuron("h2", Activation.SIGMOID, {"x1": -k, "x2": -k}, 1.5 * k)
    h3 = Neuron("h3", Activation.SIGMOID, {"x3": k}, -0.5 * k)
    h4 = Neuron("h4", Activation.SIGMOID, {"h1": k, "h2": k}, -1.5 * k)
    # h5 = Neuron("h5", Activation.SIGMOID, {"x3": 100}, -100)
    h6 = Neuron("h6", Activation.SIGMOID, {"h3": k}, -0.5 * k)
    h7 = Neuron("h7", Activation.SIGMOID, {"h4": -k, "h6": k}, -0.5 * k)
    # h8 = Neuron("h8", Activation.SIGMOID, {}, -100)
    h9 = Neuron("h9", Activation.SIGMOID, {"h4": k, "h6": -k}, -0.5 * k)
    h10 = Neuron("target", Activation.SIGMOID, {"h7": k, "h9": k}, -0.5 * k)

    neurons = [h1, h2, h3, h4, h6, h7, h9, h10]
    print("hello")
    n_nodes = 10
    neuron_graph = NeuronGraph(neurons[:n_nodes])
    data = possible_data(keys)
    q_neuron_graph = QuantizedNeuronGraph.from_neuron_graph(neuron_graph, data)
    q_neuron_graph2 = QuantizedNeuronGraph.from_neuron_graph(neuron_graph, data)
    final_ng = normalized(q_neuron_graph2)
    # bool_graph = BooleanGraph.from_q_neuron_graph(q_neuron_graph)

    ng_out = neuron_graph(data)
    ng_pred = np.where(ng_out >= 0.5, 1.0, 0.0)
    q_ng_out = q_neuron_graph(data)
    q_ng_pred = np.where(q_ng_out >= 0.5, 1.0, 0.0)
    q_ng2_out = q_neuron_graph2(data)
    q_ng2_pred = np.where(q_ng2_out >= 0.5, 1.0, 0.0)
    final_out = final_ng(data)
    final_pred = np.where(final_out >= 0.5, 1.0, 0.0)
    print(f"{ng_out = }")
    print(f"{ng_pred = }")
    print(f"{q_ng_out = }")
    print(f"{q_ng_pred = }")
    print(f"{q_ng2_out = }")
    print(f"{q_ng2_pred = }")
    print(f"{final_out = }")
    print(f"{final_pred = }")

    fidelity = np.mean(ng_pred == q_ng_pred)
    fidelity2 = np.mean(ng_pred == q_ng2_pred)
    error1 = np.mean(np.abs(ng_out - q_ng_out))
    error2 = np.mean(np.abs(ng_out - q_ng2_out))
    error3 = np.mean(np.abs(ng_out - final_out))
    print(f"error old_ng: {error1}")
    print(f"error new_ng: {error2}")
    print(f"error new ng after normalization: {error3}")

    # print(f"n_nodes: {n_nodes} | fidelity: {n_correct / n_total}")
    # print(str(bool_graph))
    # TODO: fix fidelity: it has to be 1.0 in this example
    # _ = input("Continue?")
    data = possible_data(h1.ins, is_float=True)
    y, q_neuron = from_neuron(h1, data)
    print(q_neuron)


if __name__ == "__main__":
    main()
