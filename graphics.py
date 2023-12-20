from collections.abc import Sequence
from itertools import combinations

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from neuron import Act, InputNeuron, Neuron
from utilities import set_seed

sns.set()


def possible_sums(vals: Sequence[float]) -> np.ndarray:
    """
    Given n different float values, returns a list of length 2**n, consisting
    of each value that can be produced as a sum of a subset of values in vals.
    Comparable to the powerset function, but where the subsets are a sum of their elements instead of a set that includes the elements.
    """
    n = len(vals)
    for i in range(2**n):
        pass
    ans = np.empty((2**n))
    temp = [combinations(range(n), i) for i in range(n + 1)]
    temp = [val for row in temp for val in row]

    for idx, combination in enumerate(temp):
        ans[idx] = sum(vals[c] for c in combination)
    np.ndarray.sort(ans)
    return ans


def plot_neuron_dist(neuron: Neuron) -> None:
    sums = possible_sums([val for _, val in neuron.neurons_in]) + neuron.bias
    neuron.activation
    fig, axes = plt.subplots(
        nrows=2, ncols=2, sharex=False, sharey=True, figsize=(10, 8)
    )
    x_min, x_max = sums[0], sums[-1]
    x = np.arange(x_min, x_max, 0.001)
    y = np.tanh(x)
    sns.lineplot(ax=axes[0, 0], x=x, y=y, color="r", linewidth=1.0)
    sns.scatterplot(ax=axes[0, 0], x=sums, y=np.tanh(sums), c="black", marker="X", s=50)
    # sns.histplot(sums, bins=12)
    # sns.lineplot(ax=axes[0, 0], x=x, y=y)
    fig.suptitle(f"{str(neuron)}")
    axes[0, 0].set_xlabel("s(x)")
    axes[0, 0].set_ylabel("tanh(s(x))")
    axes[0, 0].set_ylim((-1.1, 1.1))
    sns.histplot(ax=axes[1, 0], x=sums, kde=True, bins=12, stat="density")
    sns.histplot(ax=axes[0, 1], y=np.tanh(sums), kde=True, bins=24, stat="density")
    axes[0, 1].invert_xaxis()
    plt.show()


def main():
    seed = 1
    set_seed(seed)
    n = 6
    scale = 2.0
    loc = 0.5

    rand_vals = np.random.normal(loc=loc, scale=scale, size=n)
    print(rand_vals)

    neuron = Neuron(
        "y", [(InputNeuron(f"x{i + 1}"), rand_vals[i]) for i in range(n)], 0.0, Act.TANH
    )
    vals = [-1.55, -1.54, 1.6]
    neuron2 = Neuron(
        "y2",
        [(InputNeuron(f"x{i + 1}"), val) for i, val in enumerate(vals)],
        0.86,
        Act.TANH,
    )
    plot_neuron_dist(neuron2)


if __name__ == "__main__":
    main()
