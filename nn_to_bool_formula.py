import copy
import random
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bool_formula import PARITY, Bool, Interpretation
from dataloading import FileDataset
from gen_data import generate_data
from my_logging.loggers import LogMetrics, Tracker
from neuron import InputNeuron, NeuronGraph
from train_model import training_loop
from utilities import acc


class BooleanGraph(Bool):
    def __init__(self, ng: NeuronGraph) -> None:
        super().__init__()
        self.ng = ng
        self.bools = {n.name: n.to_bool() for n in self.ng.neurons}

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


def full_circle(
    target_func: Bool, model, n_datapoints=4000, epochs=5, seed=None, verbose=False
) -> Dict[str, Any]:
    layer_1 = model[0]
    assert isinstance(layer_1, nn.Linear)
    _, shape_in = layer_1.weight.shape

    # generate data for function
    vars = sorted(list(target_func.all_literals()))
    if shape_in != len(vars):
        raise ValueError(
            f"The model's input shape must be same as the number of variables in target_func, but got: {len(vars) = },  {shape_in = }"
        )
    data = generate_data(n_datapoints, target_func, vars=vars, seed=seed)

    # save it in a throwaway folder
    folder_path = Path("unittests/can_delete")
    data_path = folder_path / "gen_data.csv"
    data.to_csv(data_path, index=False, sep=",")

    # train a neural network on the dataset
    dataset = FileDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-6)
    tracker = Tracker()
    if verbose:
        tracker.add_logger(
            LogMetrics(["timestamp", "epoch", "train_loss", "train_acc"], log_every=350)
        )
    losses = training_loop(
        model,
        loss_fn,
        optim,
        dataloader,
        dataloader,
        epochs=epochs,
        tracker=tracker,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.train(False)

    # transform the trained neural network to a directed graph of perceptrons
    neuron_graph = NeuronGraph(vars, model)
    # transform the output perceptron to a boolean function
    bool_graph = BooleanGraph(neuron_graph)

    # return the found boolean function
    return {
        "vars": vars,
        "neural_network": model,
        "losses": losses,
        "dataloader": dataloader,
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


def test_model(seed, target_func, model) -> tuple[float, float, float]:
    ans = full_circle(target_func, model, epochs=200, seed=seed, verbose=False)
    bg = ans["bool_graph"]
    dl = ans["dataloader"]
    vars = ans["vars"]

    fid, bg_acc = fidelity(vars, model, bg, dl)
    nn_acc = acc(model, dl, torch.device("cpu"))
    return fid, bg_acc, nn_acc


def plot_metrics():
    n_vars = 10
    vars = [f"x{i + 1}" for i in range(n_vars)]
    target_func = PARITY(vars)
    n = len(target_func.all_literals())

    activation_sig = []
    activation_tanh = []

    best_bg_acc, best_bg_fid, best_bg_nn_acc = 0, 0, 0
    best_bg = None

    for i in range(50):
        print(f"\t------ {i} ------")
        for act in ["tanh"]:
            print(f"------ {act} ------")
            if act == "sigmoid":
                model = nn.Sequential(
                    nn.Linear(n, n),
                    nn.Sigmoid(),
                    nn.Linear(n, n - 4),
                    nn.Sigmoid(),
                    nn.Linear(n - 4, n - 4),
                    nn.Sigmoid(),
                    nn.Linear(n - 4, 1),
                    nn.Sigmoid(),
                    nn.Flatten(0),
                )
            else:
                model = nn.Sequential(
                    nn.Linear(n, n - 4),
                    nn.Tanh(),
                    nn.Linear(n - 4, n - 4),
                    nn.Tanh(),
                    nn.Linear(n - 4, n - 7),
                    nn.Tanh(),
                    nn.Linear(n - 7, 1),
                    nn.Sigmoid(),
                    nn.Flatten(0),
                )

            ans = full_circle(target_func, model, epochs=350, seed=seed, verbose=True)
            bg = ans["bool_graph"]
            dl = ans["dataloader"]
            vars = ans["vars"]

            fid, bg_acc = fidelity(vars, model, bg, dl)
            nn_acc = acc(model, dl, torch.device("cpu"))

            # Fidelity: the percentage of test examples for which the classification made by the rules agrees with the neural network counterpart
            # Accuracy: the percentage of test examples that are correctly classified by the rules
            # Consistency: is given if the rules extracted under different training sessions produce the same classifications of test examples
            # Comprehensibility: is determined by measuring the number of rules and the number of antecedents per rule
            if act == "sigmoid":
                activation_sig.append((nn_acc, bg_acc, fid))
            else:
                activation_tanh.append((nn_acc, bg_acc, fid))
            if best_bg_acc < bg_acc:
                best_bg = bg
                best_bg_nn_acc = nn_acc
                best_bg_fid = fid
                best_bg_acc = bg_acc

    print(f"{best_bg = }")
    print(f"{best_bg_nn_acc = }")
    print(f"{best_bg_fid = }")
    print(f"{best_bg_acc = }")

    sig_nn_acc = [val[0] for val in activation_sig]
    sig_bg_acc = [val[1] for val in activation_sig]
    sig_fid = [val[2] for val in activation_sig]

    tanh_nn_acc = [val[0] for val in activation_tanh]
    tanh_bg_acc = [val[1] for val in activation_tanh]
    tanh_fid = [val[2] for val in activation_tanh]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sig_nn_acc, sig_bg_acc, sig_fid, c="r", label="Sigmoid")
    ax.scatter(tanh_nn_acc, tanh_bg_acc, tanh_fid, c="g", label="Tanh")
    ax.set_xlabel("NN Accuracy")
    ax.set_ylabel("BG Accuracy")
    ax.set_zlabel("Fidelity")
    plt.legend()
    plt.show()


def knowledge_base(data: DataLoader, architecture: nn.Sequential, seed: int):
    # TODO
    pass


if __name__ == "__main__":
    # set seed to some integer if you want determinism during training
    seed: Optional[int] = 93980824973900
    # 32697229636700

    if seed is None:
        seed = torch.random.initial_seed()
    else:
        torch.manual_seed(seed)
    random.seed(seed)
    print(f"{seed = }")

    n = 10
    model = nn.Sequential(
        nn.Linear(n, n - 4),
        nn.Tanh(),
        nn.Linear(n - 4, n - 4),
        nn.Tanh(),
        nn.Linear(n - 4, 1),
        nn.Sigmoid(),
        nn.Flatten(0),
    )
    plot_metrics()

    target_func = PARITY([f"x{i + 1}" for i in range(n)])
    ans = full_circle(target_func, model, epochs=10, seed=0, verbose=False)
    bg = ans["bool_graph"]
    dl = ans["dataloader"]
    vars = ans["vars"]
    print(bg)
    fid, bg_acc = fidelity(vars, model, bg, dl)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nn_acc = acc(model, dl, device)
    print(nn_acc, fid, bg_acc)
    print(bg.ng.neurons)
