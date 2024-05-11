from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_approx(x):
    return np.where(x >= 0, 1, 0)


def tanh(x):
    return np.tanh(x)


def tanh_approx(x):
    return np.where(x >= 0, 1, -1)


def main():
    arr = np.arange(-10, 10, 0.01)

    data = {
        "x": arr,
        "sigmoid_arr": sigmoid(arr),
        "sigmoid_approx_arr": sigmoid_approx(arr),
        "tanh_arr": tanh(arr),
        "tanh_approx_arr": tanh_approx(arr),
    }

    df = pd.DataFrame(data=data)

    # sns.set_theme(style="darkgrid")
    plt.rcParams["figure.figsize"] = [14.00, 7.00]
    plt.rcParams["figure.autolayout"] = True
    fig, axes = plt.subplots(1, 2, sharey=True)

    for i in range(2):
        axes[i].axhline(0, color="black", alpha=0.5, linestyle="--")
        axes[i].axvline(0, color="black", alpha=0.5, linestyle="--")
        axes[i].set_ylabel("f(x)")

    sns.lineplot(
        data=df, x="x", y="sigmoid_arr", ax=axes[0], linewidth=2, label=r"$sigmoid(x)$"
    )
    sns.lineplot(
        data=df,
        x="x",
        y="sigmoid_approx_arr",
        ax=axes[0],
        linewidth=2,
        label=r"$sigmoid(x)$ approx.",
    )
    sns.lineplot(
        data=df, x="x", y="tanh_arr", ax=axes[1], linewidth=2, label=r"$tanh(x)$"
    )
    sns.lineplot(
        data=df,
        x="x",
        y="tanh_approx_arr",
        ax=axes[1],
        linewidth=2,
        label=r"$tanh(x)$ approx.",
    )
    plt.show()

    # ----------------------------------------------------------
    plt.subplots(nrows=2, ncols=2)

    print("Done.")


if __name__ == "__main__":
    main()
