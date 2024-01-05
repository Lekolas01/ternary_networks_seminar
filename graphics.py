import numpy as np
import seaborn as sns
from ckmeans_1d_dp import ckmeans
from matplotlib import pyplot as plt

from neuron import Activation, Neuron, QuantizedNeuron, possible_sums
from utilities import flatten, set_seed

sns.set()


def plot_neuron_dist(neuron: Neuron) -> None:
    sums = np.array(possible_sums(val for val in neuron.ins.values())) + neuron.bias
    fig, axes = plt.subplots(
        nrows=2, ncols=2, sharex=False, sharey=True, figsize=(10, 8)
    )
    margin = 0.1
    x_min, x_max = sums[0], sums[-1]
    x = np.linspace(start=x_min - margin, stop=x_max + margin, num=100)
    y = np.tanh(x)
    sns.lineplot(ax=axes[0, 0], x=x, y=y, color="r", linewidth=1.0)
    sns.scatterplot(ax=axes[0, 0], x=sums, y=np.tanh(sums), c="black", marker="X", s=50)
    q_neuron = QuantizedNeuron.from_neuron(neuron)
    fig.suptitle(f"{str(neuron)}\n{str(q_neuron)}")
    axes[0, 0].set_xlabel("s(x)")
    axes[0, 0].set_ylabel("tanh(s(x))")
    axes[0, 0].set_ylim((-1 - margin, 1 + margin))
    sns.histplot(ax=axes[1, 0], x=sums, kde=True, bins=12, stat="density")
    sns.histplot(ax=axes[0, 1], y=np.tanh(sums), kde=True, bins=24, stat="density")
    axes[0, 1].invert_xaxis()
    # for now, we only allow either 1 or 2 clusters
    ans = ckmeans(x=np.tanh(sums), k=(1, 2))
    clusters = ans.cluster
    assert all(
        clusters[i] <= clusters[i + 1] for i in range(len(clusters) - 1)
    ), "clusters must be sorted by their mean."
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
    sns.lineplot(ax=axes[1, 1], x=x_vals, y=y_vals)
    sns.lineplot(ax=axes[1, 1], x=x, y=y, color="r", linewidth=1.0)
    sns.scatterplot(ax=axes[1, 1], x=np.arctanh(centers), y=centers)
    sns.scatterplot(ax=axes[1, 1], x=sums, y=np.tanh(sums), c="black", marker="X", s=50)
    plt.show()


def main():
    seed = 1
    set_seed(seed)
    n = 6
    scale = 2.0
    loc = 0.5

    rand_vals = np.random.normal(loc=loc, scale=scale, size=n)
    print(rand_vals)

    k = 3.0
    h1 = Neuron("h1", Activation.SIGMOID, {"x1": k, "x2": k}, -0.6 * k)
    q_neuron = QuantizedNeuron.from_neuron(h1)
    plot_neuron_dist(h1)


if __name__ == "__main__":
    main()
