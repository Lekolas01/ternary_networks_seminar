import numpy as np
import seaborn as sns
from ckmeans_1d_dp import ckmeans
from matplotlib import pyplot as plt

from bool_formula import possible_data
from neuron import Activation, Neuron, NeuronGraph, possible_sums
from q_neuron import QuantizedNeuronGraph2
from utilities import flatten, set_seed

sns.set()


def plot_neuron_dist(neuron: Neuron, data=None) -> None:
    ng = NeuronGraph([neuron])
    if data is None:
        data = possible_data(neuron.ins)
    q_ng = QuantizedNeuronGraph2.from_neuron_graph(ng, data)

    sums = np.array(possible_sums(neuron.ins.values())) + neuron.bias

    fig, ax = plt.subplots()
    sns.histplot(ax=ax, x=sums, kde=True, bins=16, stat="density")
    plt.show()

    fig, axes = plt.subplots(
        nrows=1, ncols=1, sharex=False, sharey=True, figsize=(15, 5)
    )
    margin = 0.1
    data_y = neuron.act_fn(sums)
    x_min, x_max = min(sums), max(sums)
    x = np.linspace(start=x_min - margin, stop=x_max + margin, num=100)
    y = neuron.act_fn(x)
    sns.lineplot(ax=axes[1], x=x, y=y, color="r", linewidth=1.0)
    sns.scatterplot(ax=axes[1], x=sums, y=data_y, c="black", marker="X", s=50)
    # line = np.linalg.lstsq(sums, data_y)

    fig.suptitle(f"{str(neuron)}\n")
    axes[1].set_xlabel("s(x)")
    axes[1].set_ylabel("a(s(x))")
    axes[1].set_ylim((-1 - margin, 1 + margin))
    sns.histplot(ax=axes[0], x=sums, kde=True, bins=24, stat="density")
    sns.histplot(ax=axes[2], y=data_y, kde=True, bins=24, stat="density")
    # for now, we only allow either 1 or 2 clusters
    ans = ckmeans(x=data_y, k=(1, 2))
    clusters = ans.cluster
    cl_min_max: list[tuple[float, float]] = []
    for cl in clusters:
        cl_min_max.append((1, 1))

    centers = np.array([c for c in ans.centers if c != 0])
    y_thrs = (centers[:-1] + centers[1:]) / 2
    x_thrs = np.arctanh(y_thrs)
    eps = 1e-5  # need to move x thresholds a little bit, so it displays correctly
    x_vals = [x_min] + flatten([[t - eps, t + eps] for t in x_thrs]) + [x_max]
    y_vals = [y_val for y_val in centers for i in range(2)]

    if len(centers) == 1:
        print(f"constant function at neuron {neuron}.")
    sns.lineplot(ax=axes[1], x=x_vals, y=y_vals)
    sns.lineplot(ax=axes[1], x=x, y=y, color="r", linewidth=1.0)
    plt.show()


def main():
    seed = 1
    set_seed(seed)
    weights = [5.1, 5.0, 1.0, 1.0, 1.0, 0.9, 0.9]
    # weights = [0.5, 0.3, -0.7, 0.2, -0.1, 0.3, 0.4]

    h1 = Neuron(
        "target",
        Activation.TANH,
        {f"x{i + 1}": weight for i, weight in enumerate(weights)},
        -7.5,
    )
    plot_neuron_dist(h1)


if __name__ == "__main__":
    main()
