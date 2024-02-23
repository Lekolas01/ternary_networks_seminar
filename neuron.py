import copy
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from itertools import chain, combinations
from typing import Dict, TypeVar

import numpy as np
import torch
import torch.nn as nn

from bool_formula import *
from node import Graph, Node

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

    def __repr__(self):
        return str(self)


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

    def __repr__(self):
        return str(self)


def to_vars(t: torch.Tensor, names: list[str]) -> Dict[str, float]:
    t_list = t.tolist()
    assert len(t_list) == len(names)
    return {names[i]: t_list[i] for i in range(len(t_list))}
