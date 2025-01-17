import json
import math
import random
from collections.abc import Iterable
from time import sleep
from typing import TypeVar

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.types import Device
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def plot_nn_dist(model: nn.Sequential):
    params = [p.view(-1).tolist() for p in model.parameters() if p.dim() == 2]
    n_plots = len(params)
    fig, axs = plt.subplots(1, n_plots, sharex=True)
    for i in range(n_plots):
        sns.histplot(params[i], ax=axs[i])
    plt.show()


def print_metrics(alg: str, problem: str):
    with open(f"output2/{alg}/{problem}.json", "r") as f:
        t = json.load(f)
        print(f"mean_acc = {round(sum(t['acc']) / len(t['acc']), 2)}")
        print(f"std_acc = {round(np.std(t['acc']), 2)}")
        print(
            f"mean_compl = {int(round(sum(t['complexity']) / len(t['complexity']), 0))}"
        )
        print(f"std_compl = {int(round(np.std(t['complexity']), 0))}")
        print(f"best_model_acc: {round(t['best_model_acc'], 2)}")
        print(f"best_model_compl: {round(t['best_model_compl'], 0)}")


def accuracy(model: nn.Module, data_loader: DataLoader, device: Device) -> float:
    """
    Compute the accuracy of a model on the data held by a data loader.
    """
    correct_pred = 0.0
    n = 0
    if next(model.parameters()).device != device:
        model = model.to(device)

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            y_hat = model(X)
            if y_hat.dim() == 1:
                predicted_labels = torch.round(y_hat)
            else:
                _, predicted_labels = torch.max(y_hat, 1)
            correct_pred += (predicted_labels == y_true).sum()
            n += y_true.size(0)
    return float(correct_pred) / n


def distance_from_int_precision(x: np.ndarray):
    """
    Calculates the avg. distance of x w.r.t. the nearest integer.
    x must be in range [-1, 1]
    When doing many update steps with WDR Regularizer, this distance should decrease.
    """
    assert np.allclose(x, np.zeros_like(x), atol=1, rtol=0)
    ans = np.full_like(x, 2)
    bases = [-1, 0, 1]
    for base in bases:
        d = np.abs(x - base)
        ans = np.where(d < ans, d, ans)
    minus_ones = np.count_nonzero(x <= -0.5) / x.shape[0]
    plus_ones = np.count_nonzero(x >= 0.5) / x.shape[0]
    zeros = 1 - minus_ones - plus_ones
    return np.mean(ans), zeros


def get_all_weights(model: nn.Module):
    params = [param.view(-1) for param in model.parameters()]
    params = torch.cat(params)
    return params


def set_seed(seed: int | None) -> int:
    """Call this function if you need determinism."""

    if seed is None:
        seed = torch.random.initial_seed()
    else:
        torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


T = TypeVar("T")


def flatten(l: Iterable[Iterable[T]]) -> list[T]:
    return [x for row in l for x in row]


def invert_dict(d: dict) -> dict:
    ans = {}
    for k, v in d.items():
        for x in v:
            ans.setdefault(x, []).append(k)
    return ans


def progress_bar(progress: int, total: int, width=40) -> str:
    bars = ["0x2588"]
    rel_progress = progress / float(total)
    rel_width = width * rel_progress
    left_bars = int(rel_width)
    last_eights = int(8 * (rel_width % 1))
    val = 9615 - last_eights
    last_bar = chr(val)  # TODO
    bar = chr(9608) * left_bars + last_bar + "-" * (width - int(rel_width))
    written_progress = f"{str(progress).rjust(len(str(total)))} / {total}"
    return f"\r|{bar}|{written_progress}"


def plot_1d_dist(x, bins=None):
    plt.hist(x, bins)
    plt.show()


def main():
    # for i in tqdm(range(100), desc="Loading...", ascii=False):
    #     sleep(0.05)
    # print()

    problems = [
        "abcdefg",
        "balance-scale",
        "car-evaluation",
        "king-rook-king-pawn",
        "king-rook-king",
        "monk-1",
        "monk-2",
        "monk-3",
        "mushroom",
        "parity10",
        "tic-tac-toe",
        "vote",
    ]
    for problem in problems:
        print(f"----------- {problem} -----------")
        for alg in ["dre", "ripper"]:
            print(f"\t{alg}:")
            try:
                print_metrics(problem, alg)
            except:
                print(f"Could not find files for {problem}")


if __name__ == "__main__":
    main()
