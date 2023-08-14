from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from bool_formula import *
import numpy as np


class Neuron:
    def __init__(
        self,
        name="",
        neurons_in: list[tuple[Neuron, float]] = [],
        bias: float = 0.0,
    ) -> None:
        self.neurons_in = neurons_in
        self.bias = bias
        self.name = name

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
            input_neurons: list[tuple[Neuron, float]],
            bias: float,
            i: int = 0,
        ) -> Boolean:
            if bias <= 0:
                return Constant(True)
            if i == len(input_neurons):
                return Constant(False)

            name = input_neurons[i][0].name
            weight = input_neurons[i][1]
            # set to False
            term1 = to_bool_rec(input_neurons, bias, i + 1)
            term2 = Func(
                Op.AND,
                [to_bool_rec(input_neurons, bias - weight, i + 1), Literal(name, True)],
            )
            if any(term == Constant(True) for term in [term1, term2]):
                return Constant(True)
            return Func(Op.OR, [term1, term2])

        def simplified(b: Boolean, layer=0) -> Boolean:
            if isinstance(b, Func):
                for i, child in enumerate(b.children):
                    b.children[i] = simplified(child, layer + 1)
                if b.bin_op == Op.AND:
                    # if one term is False, the whole conjunction is False
                    if any(child == Constant(False) for child in b.children):
                        return Constant(False)

                    # True constants can be removed
                    b.children = list(filter(lambda c: c != Constant(True), b.children))
                    # if now the list is empty, we can return True
                    if len(b.children) == 0:
                        return Constant(True)
                    # otherwise return b
                if b.bin_op == Op.OR:
                    # if one term is True, the whole conjunction is True
                    if any(child == Constant(True) for child in b.children):
                        return Constant(True)

                    # False constants can be removed
                    b.children = list(
                        filter(lambda c: c != Constant(False), b.children)
                    )
                    # if now the list is empty, we can return False
                    if len(b.children) == 0:
                        return Constant(False)
                    # otherwise return b
                if len(b.children) == 1:
                    return b.children[0]
            return b

        # sort neurons by their weight
        neurons_in = sorted(self.neurons_in, key=lambda x: x[1], reverse=True)
        # TODO: also allow negative weights
        assert all(
            t[1] >= 0 for t in neurons_in
        ), "Not yet implemented for negative numbers."
        new_var = to_bool_rec(neurons_in, self.bias)

        print(new_var)
        ans = simplified(new_var)
        return ans


if __name__ == "__main__":
    x1 = Neuron("x1")
    x2 = Neuron("x2")
    x3 = Neuron("x3")
    x4 = Neuron("x4")
    b = Neuron("b", [(x1, 1.2), (x2, 3.0), (x3, 1.6), (x4, 0.3)], 2.2)
    print(b.to_bool())
    