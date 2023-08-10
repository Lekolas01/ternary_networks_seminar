from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from enum import Enum
import numpy.typing as npt


class Literal:
    def __init__(self, value: object, positive: bool) -> None:
        assert value is not None
        self.value = value
        self.positive = positive

    def __str__(self) -> str:
        return f"{'!' if not self.positive else ''}{str(self.value)}"


class BinaryOp(Enum):
    AND = "&"
    OR = "|"


class Tree:
    def __init__(self, val: Literal | BinaryOp, children: list = []) -> None:
        self.val = val
        self.children = children

    def __str__(self) -> str:
        if isinstance(self.val, Literal):
            return str(self.val)
        elif isinstance(self.val, Tree):
            op = str(self.val.value)
            return f"({f' {op} '.join([str(c) for c in self.children])})"
        else:
            raise ValueError(
                f"Error for argument val: Expected Literal | BinaryOp, but got {type(self.val)}."
            )


def sum_2_dnf(weights: np.ndarray, bias: float):
    """
    Create a formula in DNF that is logically equivalent to the sign activation function applied to the linear combination with some weights.
    For example: [2.5*x1 + 3.0*x2 >= 4.0] -> [x1 AND x2]
    """
    ans = []
    assert len(weights) >= 1
    n = len(weights)
    # order = weights.argsort()
    # weights = weights[order]
    vals = np.zeros(n)
    n_iter = 2**n
    for _ in range(n_iter):
        sum = np.sum(weights * vals)
        if sum + bias >= 0:
            ans.append(vals.copy())
        idx = n - 1
        while vals[idx] == 1:
            vals[idx] = 0
            idx -= 1
        vals[idx] = 1
    return np.array(ans)


def terms_2_dnf(terms: np.ndarray, names: list[str]) -> list[str]:
    def val_2_str(term: np.ndarray, names: list[str]) -> str:
        return "".join([names[i] if term[i] == 1 else "" for i in range(len(term))])

    return [val_2_str(terms[i], names) for i in range(terms.shape[0])]


def nn_2_bool_formula(model: nn.Module) -> Optional[Tree]:
    """ """
    for name, module in model.named_children():
        print(f"{name} = {module}")
        if isinstance(module, nn.Sequential):
            for layer_name, layer in module.named_children():
                if isinstance(layer, nn.Linear):
                    sum_2_dnf(layer.weight[0, :], layer.bias[0])
    # get the weights to the first node in the second layer
    # weights =
    pass


if __name__ == "__main__":
    path = Path("runs/logical_AND/best/config01_epoch001.pth")
    model = torch.load(path)
    # formula = nn_2_bool_formula(model)
    # print(f"The converted model is {formula}")
    weights = np.array([2.4, 2.1, 1.8, 1.3, 1.1, 0.8])
    terms = sum_2_dnf(weights, -3.5)
    terms.sort(axis=1)
    print(terms_2_dnf(terms, names=[f"x{i}" for i in range(1, len(weights) + 1)]))
