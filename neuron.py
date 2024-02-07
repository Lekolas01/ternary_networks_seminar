import bisect
import copy
import functools
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from copy import deepcopy
from enum import Enum
from graphlib import TopologicalSorter
from itertools import chain, combinations
from typing import Dict, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn
from ckmeans_1d_dp import ckmeans
from numpy import ndarray

from bool_formula import *
from node import Graph, Node
from utilities import flatten

Val = TypeVar("Val")


def bool_2_ch(x: bool) -> str:
    return "T" if x else "F"


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

    def act_fn(self, x) -> np.ndarray:
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
        varnames = list(copy.copy(varnames))
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
    def from_neuron_graph(
        cls, ng: NeuronGraph, data: MutableMapping[str, np.ndarray], verbose=False
    ):
        new_ng = deepcopy(ng)
        q_neuron_graph = QuantizedNeuronGraph([])
        graph_outs = new_ng.graph_outs()
        graph_ins = new_ng.graph_ins()
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


class DpNode:
    def __init__(self, key: str, min_thr: float, max_thr: float) -> None:
        self.key = key
        self.min_thr = min_thr
        self.max_thr = max_thr
        self.mean = (min_thr + max_thr) / 2

    def __repr__(self) -> str:
        return f"DpNode({self.key}, {self.min_thr}, {self.max_thr})"


class Dp:
    def __init__(self, n_vars: int) -> None:
        assert 0 <= n_vars
        self.n_vars = n_vars
        self.data: list[list[DpNode]] = [[] for _ in range(self.n_vars + 1)]
        self.eps = 1e-10

    def find(self, k: int, val: float) -> DpNode | None:
        assert k >= 0, f"k must be >= 0, but got {k}."
        if k > len(self.data):
            return None
        arr = self.data[k]
        for t in arr:
            if t.min_thr + self.eps < val < t.max_thr - self.eps:
                return t
        return None

    def insert(self, k: int, val: DpNode):
        # assert you don't add part of a range that is already included
        if (
            self.find(k, val.min_thr) is not None
            or self.find(k, val.max_thr) is not None
        ):
            print(f"{self.data = }")
            print(f"{k = }")
            print(f"{val = }")
            raise ValueError

        bisect.insort(self.data[k], val, key=lambda x: x.min_thr)

    def __getitem__(self, key: int):
        assert 0 <= key <= self.n_vars
        return self.data[key]

    def __str__(self) -> str:
        ans = []
        for i in range(self.n_vars + 1):
            ans.append("[" + ", ".join(str(t) for t in self.data[i]) + "]")
        return "\n".join(ans)

    def __repr__(self) -> str:
        return str(self)

    def __len__(self):
        return len(self.data)


class IfThenRule(Node):
    def __init__(
        self, key: str, ins: list[Tuple[str, bool]], val: bool | None = None
    ) -> None:
        self.key = key
        self.ins = ins
        self.val = val
        self.is_const = self.val is not None

    def __call__(self, vars: Mapping[str, ndarray]) -> np.ndarray:
        if self.is_const:
            return np.array(True) if self.val else np.array(False)

        key = self.ins[0][0]
        ans = np.ones_like(vars[key], dtype=bool)
        for name, val in self.ins:
            temp = vars[name] if val else ~vars[name]
            try:
                ans = ans & temp
            except:
                temp = 0
        return ans

    def simplify(self, knowledge: Knowledge) -> bool:
        if self.is_const:
            assert isinstance(self.val, bool)
            ans = not (self.key in knowledge)
            knowledge[self.key] = Constant(self.val)
            return ans
        changed = False
        to_delete_ins = []

        for idx, (lit, lit_val) in enumerate(self.ins):
            if lit in knowledge:
                if isinstance(knowledge[lit], Constant):
                    new_lit_val = lit_val if knowledge[lit]() else not lit_val
                    if new_lit_val:
                        # delete any positive constants
                        to_delete_ins.append(lit)
                        changed = True
                    else:
                        # if you have a negative constant, the whole rule is negative
                        self.is_const = True
                        self.val = False
                        return True
                elif isinstance(knowledge[lit], Literal):
                    self.ins[idx] = (str(knowledge[lit]), lit_val)
                elif isinstance(knowledge[lit], NOT):
                    c = knowledge[lit].child  # type: ignore
                    self.ins[idx] = (str(c), not lit_val)

                # elif isinstance(knowledge[lit], NOT):
                # self.ins[idx] = (str(knowledge[lit]), lit_val)
        self.ins = [t for t in self.ins if t[0] not in to_delete_ins]
        # if there are no literals left, i.e. every literal is positive, the whole rule is positive
        if len(self.ins) == 0:
            self.is_const = True
            self.val = True
            return True
        return changed

    def __repr__(self) -> str:
        return f"{self.key} := {self.body()}"

    def __str__(self) -> str:
        return self.__repr__()

    def body(self) -> str:
        if self.is_const:
            assert isinstance(self.val, bool)
            return bool_2_ch(self.val)
        return ", ".join(f"{'' if b else '!'}{key}" for key, b in self.ins)

    def children(self) -> list[str]:
        if self.is_const:
            return []
        return [key for key, _ in self.ins]


class Subproblem:
    def __init__(self, key: str, rules: list[IfThenRule]):
        self.key = key
        self.rules = rules
        self.is_const = False
        self.val = False

    def __call__(self, vars: MutableMapping[str, np.ndarray]) -> np.ndarray:
        if self.is_const:
            return np.array(self.val)
        temp = [rule(vars) for rule in self.rules]
        return functools.reduce(lambda x, y: x | y, temp)

    def simplify(self, knowledge: Knowledge) -> bool:
        """
        Returns a boolean that tells whether something changed
        """
        # first, simplify each individual rule.
        # if no rule changed, the whole subproblem did not change, -> return False.
        changed = False
        for rule in self.rules:
            temp = rule.simplify(knowledge)
            changed = changed or temp

        # filter all constant F rules, as they will never trigger
        self.rules = [r for r in self.rules if not (r.is_const and not r.val)]

        # if there exists a constant T rule, the whole subproblem is T
        if any(r.is_const and r.val for r in self.rules):
            knowledge[self.key] = Constant(True)
            self.is_const, self.val = True, True

        # if there are no non-constant rules left, the whole subproblem is F
        if len(self.rules) == 0:
            knowledge[self.key] = Constant(False)
            self.is_const, self.val = True, False

        # if the subproblem only has one rule left with exactly one value,
        # add the value to the knowledge and delete the subproblem
        if len(self.rules) == 1 and len(self.rules[0].ins) == 1:
            t = self.rules[0].ins[0]
            v = Literal(t[0])
            if not t[1]:
                v = NOT(v)
            knowledge[self.key] = v

        """
        to further simplify subproblems, one should also look at relations between 
        the rules. For example, if one rule's body is a strict superset of another rule's
        body, you can delete the more specific rule, as it will only ever trigger when the 
        more general triggers anyways. In our use case however, that can not happen.
        """
        return changed

    def __str__(self) -> str:
        if self.is_const:
            return f"SP({self.key} := {bool_2_ch(self.val)})"
        return f"SP({self.key} := {' | '.join(r.body() for r in self.rules)})"

    def __repr__(self) -> str:
        return str(self)

    def children(self) -> list[str]:
        return flatten([rule.children() for rule in self.rules if not rule.is_const])


class RuleSetNeuron(Node):
    def __init__(self, q_neuron: QuantizedNeuron, simplify: bool) -> None:
        self.q_neuron = q_neuron
        self.key = q_neuron.key
        self.ins = q_neuron.ins
        self.dp = self.calc_dp()
        self.subproblems = self.to_subproblems(self.dp)
        self.target_node = self.call_order()[-1]
        self.knowledge = {}
        if simplify:
            # simplify rules
            self.simplify()

    def __call__(self, vars: MutableMapping[str, np.ndarray]) -> np.ndarray:
        vars = copy.copy(vars)
        for k in self.knowledge:
            vars[k] = np.array(self.knowledge[k])
        for key in self.call_order():
            sp = self.subproblems[key]
            vars[key] = sp(vars)
        return vars[self.key]

    def name_gen(self):
        yield self.key
        idx = 0
        while True:
            idx += 1
            yield f"{self.key}_{idx}"

    def calc_dp(self) -> Dp:
        def to_bool_rec(k: int, threshold: float, dp: Dp) -> DpNode:
            max_sum: float = float(sum(n[1] for n in self.n_ins[k:]))
            # how much one could add by setting every variable to 1 without changing the formula
            found = dp.find(k, threshold)
            if found is not None:
                return found

            # if already positive, return True
            if threshold >= 0.0:
                ans = DpNode("rename_me", 0.0, float("inf"))
                dp.insert(k, ans)
                return ans
            # if you can't reach positive values, return False
            if max_sum + threshold <= 0.0:
                ans = DpNode("rename_me", float("-inf"), -max_sum)
                dp.insert(k, ans)
                return ans

            weight = self.n_ins[k][1]

            # set to False
            n1 = to_bool_rec(k + 1, threshold, dp)
            n2 = to_bool_rec(k + 1, threshold + weight, dp)
            new_min, new_max = (
                max(n1.min_thr, n2.min_thr - weight),
                min(n1.max_thr, n2.max_thr - weight),
            )
            ans = DpNode("rename_me", new_min, new_max)
            dp.insert(k, ans)
            return ans

        assert self.q_neuron.y_centers == [0.0, 1.0]
        assert self.q_neuron.x_thr == 0.0
        self.bias = self.q_neuron.bias

        # sort neurons by their weight
        ins = list(self.ins.items())
        ins = sorted(ins, key=lambda x: abs(x[1]), reverse=True)

        # remember which weights are negative and then invert all of them (needed for negative numbers)
        self.signs = [tup[1] < 0 for tup in ins]
        self.n_ins = [(tup[0], abs(tup[1])) for tup in ins]
        self.n_vars = len(self.n_ins)

        positive_weights = list(zip(self.signs, [tup[1] for tup in self.n_ins]))
        filtered_weights = list(filter(lambda tup: tup[0], positive_weights))
        self.bias_diff = sum(tup[1] for tup in filtered_weights)

        dp = Dp(len(self.n_ins))
        self.ans = to_bool_rec(0, self.bias - self.bias_diff, dp)

        # give every node in the directed bool graph a name
        names = self.name_gen()
        for k in range(self.n_vars + 1):
            for node in dp[k]:
                node.key = next(names)
        return dp

    @classmethod
    def from_q_neuron(cls, q_neuron: QuantizedNeuron):
        return RuleSetNeuron(q_neuron, True)

    def __str__(self) -> str:
        ans = (
            "RuleSet[\n\t"
            + "\n\t".join(str(sp) for sp in self.subproblems.values())
            + "\n]"
        )
        return ans

    def __repr__(self) -> str:
        return str(self)

    def to_subproblems(self, dp: Dp) -> dict[str, Subproblem]:
        ans: dict[str, Subproblem] = {}
        # Operator()
        # then create 1 or 2 if-then rules for each node, depending on whether it's
        # a constant or not
        for k in range(self.n_vars + 1):
            for node in dp[k]:
                if node.min_thr == float("-inf"):
                    ans[node.key] = Subproblem(
                        node.key, [IfThenRule(node.key, [], val=False)]
                    )
                elif node.max_thr == float("inf"):
                    ans[node.key] = Subproblem(
                        node.key, [IfThenRule(node.key, [], val=True)]
                    )
                else:
                    target_1 = dp.find(k + 1, node.mean)
                    assert target_1 is not None

                    target_2 = dp.find(k + 1, node.mean + self.n_ins[k][1])
                    assert target_2 is not None
                    rule1 = IfThenRule(node.key, [(target_1.key, True)])
                    rule2 = IfThenRule(
                        node.key,
                        [
                            (self.n_ins[k][0], not self.signs[k]),
                            (target_2.key, True),
                        ],
                    )
                    ans[node.key] = Subproblem(key=node.key, rules=[rule1, rule2])
        return ans

    def graph_ins(self) -> dict[str, set[str]]:
        graph_ins = {}
        keys = {key for key in self.subproblems}
        for key, sp in self.subproblems.items():
            graph_ins[key] = list(filter(lambda k: k in keys, sp.children()))
        return graph_ins

    def call_order(self) -> list[str]:
        sorter = TopologicalSorter(self.graph_ins())
        return list(sorter.static_order())

    def simplify(self) -> None:
        changed = True
        # simplify each subproblem in topological order

        while changed:
            changed = False
            for key in self.call_order():
                temp = self.subproblems[key].simplify(self.knowledge)
                changed = changed or temp
            # filter constant subproblems
            self.subproblems = {
                key: sp for key, sp in self.subproblems.items() if not sp.is_const
            }
            # filter subproblems that are not in use anymore
            all_children = set(
                flatten(sp.children() for sp in self.subproblems.values())
            )
            all_children.add(self.target_node)
            old_len = len(self.subproblems)
            self.subproblems = {
                key: sp for key, sp in self.subproblems.items() if key in all_children
            }


class RuleSetGraph(Graph):
    def __init__(self, rule_set_neurons: Sequence[RuleSetNeuron]) -> None:
        super().__init__(rule_set_neurons)

    @classmethod
    def from_q_neuron_graph(cls, q_ng: QuantizedNeuronGraph, simplify=True):
        rule_neurons = [
            RuleSetNeuron(q_n, simplify)
            for key, q_n in q_ng.nodes.items()
            if isinstance(q_n, QuantizedNeuron)
        ]
        ans = RuleSetGraph(rule_neurons)
        return ans


def nn_to_rule_set(
    model: nn.Sequential,
    data: MutableMapping[str, np.ndarray],
    vars: Sequence[str],
    verbose=False,
):
    # transform the trained neural network to a directed graph of full-precision neurons
    neuron_graph = NeuronGraph.from_nn(model, vars)
    # transform the graph to a new graph of perceptrons with quantized step functions
    q_neuron_graph = QuantizedNeuronGraph.from_neuron_graph(
        neuron_graph, data, verbose=verbose
    )
    # transform the quantized graph to a set of if-then rules
    bool_graph = RuleSetGraph.from_q_neuron_graph(q_neuron_graph)
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
