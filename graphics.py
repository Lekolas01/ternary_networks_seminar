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

    n_bins = 21

    sns.histplot(x=sums, kde=True, bins=n_bins, stat="density")
    ans = ckmeans(x=sums, k=(2))
    centers = np.array([c for c in ans.centers if c != 0])
    x_thrs = (centers[:-1] + centers[1:]) / 2
    cluster_mean = centers.mean()
    plt.axvline(cluster_mean, -1, 2, color="green", linewidth=2, linestyle="dotted")
    plt.axvline(centers[0], -0.1, 0.03, color="black", linewidth=2)
    plt.axvline(centers[1], -0.1, 0.03, color="black", linewidth=2)
    plt.show()

    sns.histplot(x=neuron.act_fn(sums), kde=True, bins=n_bins, stat="density")

    y = neuron.act_fn(sums)
    ans = ckmeans(x=y, k=(2))
    centers = np.array([c for c in ans.centers if c != 0])
    y_thr = centers.mean()
    cluster_mean = centers.mean()
    plt.axvline(cluster_mean, -1, 2, color="green", linewidth=2, linestyle="dotted")
    plt.axvline(centers[0], -0.1, 0.03, color="black", linewidth=2)
    plt.axvline(centers[1], -0.1, 0.03, color="black", linewidth=2)

    plt.show()

    margin = 0.1
    data_y = neuron.act_fn(sums)
    x_max = max(sums)
    x_min = -x_max
    x = np.linspace(start=x_min - margin, stop=x_max + margin, num=100)
    y = neuron.act_fn(x)
    ans = ckmeans(x=data_y, k=(1, 2))
    centers = np.array([c for c in ans.centers if c != 0])
    y_thrs = (centers[:-1] + centers[1:]) / 2
    x_thrs = np.arctanh(y_thrs)

    plt.show()
    return
    sns.histplot(ax=axes[1, 1], y=data_y, kde=True, bins=24, stat="density")
    # for now, we only allow either 1 or 2 clusters
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
    weights = np.array([4.1, 3.2, 2.5, 2.1, 1.0, 0.4]) * 1.0

    h1 = Neuron(
        "target",
        Activation.TANH,
        {f"x{i + 1}": weight for i, weight in enumerate(weights)},
        -sum(weights) / 2,
    )
    plot_neuron_dist(h1)


if __name__ == "__main__":
    main()
