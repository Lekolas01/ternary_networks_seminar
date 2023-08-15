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
            neurons_in: list[tuple[Neuron, float]],
            neuron_signs: list[bool],
            threshold: float,
            i: int = 0,
        ) -> Boolean:
            if threshold < 0:
                return Constant(True)  
            if i == len(neurons_in):
                return Constant(False)

            name = neurons_in[i][0].name
            positive = not neuron_signs[i]
            weight = neurons_in[i][1]

            # set to False
            term1 = to_bool_rec(neurons_in, neuron_signs, threshold, i + 1)
            term2 = Func(
                Op.AND,
                [to_bool_rec(neurons_in, neuron_signs, threshold - weight, i + 1), Literal(name, positive)],
            )
            return Func(Op.OR, [term1, term2])
        
        # sort neurons by their weight
        neurons_in = sorted(self.neurons_in, key=lambda x: abs(x[1]), reverse=True)

        # remember which weights are negative and then invert all of them (needed for negative numbers)
        negative = [tup[1] < 0 for tup in neurons_in]
        neurons_in = [(tup[0], abs(tup[1])) for tup in neurons_in]

        positive_weights = zip(negative, [tup[1] for tup in neurons_in])
        filtered_weights = filter(lambda tup: tup[0], positive_weights)
        bias_diff = sum(tup[1] for tup in filtered_weights)
        long_ans = to_bool_rec(neurons_in, negative, -self.bias + bias_diff)

        ans = simplified(long_ans)
        return ans


if __name__ == "__main__":
    x1 = Neuron("x1")
    x2 = Neuron("x2")
    x3 = Neuron("x3")
    x4 = Neuron("x4")
    #b = Neuron("b", [(x1, 1.2), (x2, 3.0), (x3, 1.6), (x4, 0.3)], 2.2)
    b = Neuron("b", [(x1, 1.5), (x2, -1.4), (x3, 2.1), (x4, -0.3)], 1.0)
    print(b.to_bool())
    