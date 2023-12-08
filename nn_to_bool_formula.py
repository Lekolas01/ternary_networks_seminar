import copy
import random
from pathlib import Path
from typing import Any, Dict, Optional
import torch.functional as F

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bool_formula import PARITY, Bool, Interpretation
from datasets import FileDataset
from gen_data import generate_data
from my_logging.loggers import LogMetrics, LogModel, Tracker
from neuron import InputNeuron, NeuronGraph
from train_model import training_loop
from utilities import acc, plot_nn_dist, set_seed


class BooleanGraph(Bool):
    def __init__(self, ng: NeuronGraph) -> None:
        super().__init__()
        self.ng = ng
        self.bools = {}
        for n in self.ng.neurons:
            print(f"{n = }")
            temp = n.to_bool()
            self.bools[n.name] = temp

    def __call__(self, interpretation: Interpretation) -> bool:
        int_copy = copy.copy(interpretation)
        for key in self.bools:
            n_bool = self.bools[key]
            val = n_bool(int_copy)
            int_copy[key] = val

        target_name = self.ng.target().name
        return int_copy[target_name]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        ans = "BooleanGraph[\n"
        for key in self.bools:
            n_bool = self.bools[key]
            ans += f"\t{key} := {str(n_bool)}\n"
        ans += "]\n"
        return ans

    def all_literals(self) -> set[str]:
        ans = set()
        for key in self.bools:
            if isinstance(self.bools[key], InputNeuron):
                ans = ans.union(self.bools[key].all_literals())
        return ans

    def negated(self) -> Bool:
        raise NotImplementedError


def gen_dataset_from_func(func: Bool, n_datapoints: int) -> Path:
    # generate data for function
    vars = sorted(list(func.all_literals()))
    data = generate_data(func, n_datapoints, vars=vars, seed=seed)
    # save it in a throwaway folder
    folder_path = Path("unittests/can_delete")
    data_path = folder_path / "gen_data.csv"
    data.to_csv(data_path, index=False, sep=",")
    return data_path


def train_rules(
    train_dl: DataLoader,
    valid_dl: DataLoader,
    nn_model: nn.Sequential,
    epochs: int,
    vars: list[str],
) -> Dict[str, Any]:
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"{device = }")
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(nn_model.parameters(), lr=0.002, weight_decay=1e-6)
    tracker = Tracker()
    tracker.add_logger(
        LogMetrics(["timestamp", "epoch", "train_loss", "train_acc"], log_every=2)
    )
    # tracker.add_logger(LogModel(log_every=10))

    losses = training_loop(
        nn_model,
        loss_fn,
        optim,
        train_dl,
        valid_dl,
        epochs=epochs,
        lambda1=1e-4,
        tracker=tracker,
        device=device,
    )
    nn_model.train(False)

    # plot parameter distribution

    plot_nn_dist(nn_model)

    # transform the trained neural network to a directed graph of perceptrons
    neuron_graph = NeuronGraph(vars, nn_model)
    # transform the neuron graph to a boolean function
    bool_graph = BooleanGraph(neuron_graph)

    # return the found boolean function
    return {
        "vars": vars,
        "neural_network": nn_model,
        "losses": losses,
        "neuron_graph": neuron_graph,
        "bool_graph": bool_graph,
    }


def fidelity(vars, model, bg, dl) -> tuple[float, float]:
    fid = 0
    n_vals = 0
    bg_accuracy = 0
    for X, y in dl:
        nn_pred = model(X)
        nn_pred = [bool(val.round()) for val in nn_pred]

        data = []
        n_rows, n_cols = X.shape
        for i in range(n_rows):
            row = X[i]
            new_var = {vars[j]: bool(row[j]) for j in range(n_cols)}
            data.append(new_var)
        bg_pred = [bg(datapoint) for datapoint in data]
        bg_correct = [bg_pred[i] == y[i] for i in range(n_rows)]
        n_correct = len(list(filter(lambda x: x, bg_correct)))
        bg_accuracy += n_correct
        bg_same = [bg_pred[i] == nn_pred[i] for i in range(n_rows)]
        n_same = len(list(filter(lambda x: x, bg_same)))
        fid += n_same
        n_vals += n_rows
    return fid / n_vals, bg_accuracy / n_vals


def learn_parity(n_vars: int):
    assert n_vars > 1, f"n_vars must be >= 2, but got: {n_vars}"
    vars = [f"x{i + 1}" for i in range(n_vars)]
    target_func = PARITY(vars)
    n = len(target_func.all_literals())

    metrics = []

    best_bg_acc, best_bg_fid, best_bg_nn_acc = 0, 0, 0
    best_bg = None

    model = nn.Sequential(
        nn.Linear(n, 2 * n),
        nn.Tanh(),
        nn.Linear(2 * n, 2 * n),
        nn.Tanh(),
        nn.Linear(2 * n, 1),
        nn.Sigmoid(),
        nn.Flatten(0),
    )
    path = gen_dataset_from_func(target_func, n_datapoints=int(2**n))
    train_dl = DataLoader(FileDataset(path))
    valid_dl = DataLoader(FileDataset(path))
    epochs = 350
    ans = train_rules(train_dl, valid_dl, model, epochs, vars)
    bg = ans["bool_graph"]
    vars = ans["vars"]

    fid, bg_acc = fidelity(vars, model, bg, train_dl)
    nn_acc = acc(model, train_dl, torch.device("cpu"))

    # Fidelity: the percentage of test examples for which the classification made by the rules agrees with the neural network counterpart
    # Accuracy: the percentage of test examples that are correctly classified by the rules
    # Consistency: is given if the rules extracted under different training sessions produce the same classifications of test examples
    # Comprehensibility: is determined by measuring the number of rules and the number of antecedents per rule
    metrics.append((nn_acc, bg_acc, fid))
    if best_bg_acc < bg_acc:
        best_bg = bg
        best_bg_nn_acc = nn_acc
        best_bg_fid = fid
        best_bg_acc = bg_acc

    print(f"{best_bg = }")
    print(f"{best_bg_nn_acc = }")
    print(f"{best_bg_fid = }")
    print(f"{best_bg_acc = }")

    tanh_nn_acc = [val[0] for val in metrics]
    tanh_bg_acc = [val[1] for val in metrics]
    tanh_fid = [val[2] for val in metrics]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(tanh_nn_acc, tanh_bg_acc, tanh_fid, c="g", label="Tanh")
    ax.set_xlabel("NN Accuracy")
    ax.set_ylabel("BG Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 56056698460700
    seed = set_seed(56056698460700)
    print(f"{seed = }")

    n_vars = 10
    learn_parity(n_vars)

    print(f"{seed = }")
