import copy
from collections.abc import Mapping, MutableMapping

import numpy as np
from ckmeans_1d_dp import ckmeans

from bool_formula import PARITY, Activation, possible_data
from neuron import Neuron, NeuronGraph
from node import Graph, Node


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
    def from_neuron_graph(
        cls, ng: NeuronGraph, data: MutableMapping[str, np.ndarray], verbose=False
    ):
        new_ng = copy.deepcopy(ng)
        q_neuron_graph = QuantizedNeuronGraph([])
        graph_outs = new_ng.outs()
        graph_ins = new_ng.ins()
        for key, neuron in new_ng.neurons.items():
            ins = graph_ins[key]
            if key not in new_ng.out_keys:  # non-output nodes
                outs = graph_outs[neuron.key]
                data_y = neuron(data)
                assert isinstance(data_y, np.ndarray)
                data[key] = data_y

                ck_ans = ckmeans(data_y, (2))
                if verbose:
                    print(neuron)
                    print(f"{ck_ans.tot_withinss / len(data_y)}")
                    print(f"{ck_ans.totss / len(data_y)}")
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

    def __repr__(self):
        return str(self)


class QuantizedNeuronGraph2(Graph):
    def __init__(self, q_neurons: list[QuantizedNeuron]) -> None:
        super().__init__(q_neurons)

    @classmethod
    def from_neuron_graph(
        cls, ng: NeuronGraph, data: MutableMapping[str, np.ndarray], verbose=False
    ):
        new_ng = copy.deepcopy(ng)
        q_neuron_graph = QuantizedNeuronGraph2([])
        graph_outs = new_ng.outs()
        graph_ins = new_ng.ins()
        for key, neuron in new_ng.neurons.items():
            ins = graph_ins[key]
            outs = graph_outs[neuron.key]
            data_y = neuron(data)
            assert isinstance(data_y, np.ndarray)
            data[key] = data_y

            ck_ans = ckmeans(data_y, (2))
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

        return q_neuron_graph

    def __repr__(self):
        return str(self)


def normalize(q_ng: QuantizedNeuronGraph) -> QuantizedNeuronGraph:
    return q_ng


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
