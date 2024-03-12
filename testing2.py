import itertools

import numpy as np


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


def main():
    thr = 0.6
    ans = np.zeros((10, 10), dtype=int)
    for i, j in itertools.product(range(ans.shape[0]), range(ans.shape[1])):
        ans[i, j] = find_smallest_k(i + 1, j + 1, thr)
    print(ans)


if __name__ == "__main__":
    main()
