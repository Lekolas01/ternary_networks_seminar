import itertools

import numpy as np
import torch
import torch.nn as nn


def h2(n, k) -> float:
    if k > n or k < 2:
        return 0
    return k * (k - 1) / n / (n - 1)


def p(n, k, m) -> float:
    return 1 - np.power((1 - h2(n, k)), m)


def find_smallest_k(n_inputs: int, n_outputs: int, thr: float) -> int:
    if n_inputs <= 2:
        return n_inputs
    k = n_inputs
    while p(n_inputs, k, n_outputs) >= thr:
        k -= 1
    return k + 1


class FilteredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 7),
            nn.Tanh(),
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 1),
        )
        self.disable_weights()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def disable_weights(self):
        lin1 = list(self.linear_relu_stack)[0]
        assert isinstance(lin1, nn.Linear)
        lin1.weight.data
        """Set some weights to 0 and lock them, so their values can't change,
        effectively removing those edges from the computational graph."""
        for layer in self.linear_relu_stack:
            if not isinstance(layer, nn.Linear):
                continue
            layer.parameters()
        pass


def main():
    thr = 0.6
    ans = np.zeros((10, 10), dtype=int)
    for i, j in itertools.product(range(ans.shape[0]), range(ans.shape[1])):
        ans[i, j] = find_smallest_k(i + 1, j + 1, thr)
    print(ans)
    print(find_smallest_k(100, 30, 0.90))
    a = FilteredModel()


if __name__ == "__main__":
    main()
