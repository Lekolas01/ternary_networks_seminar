import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bool_formula import PARITY
from datasets import FileDataset
from gen_data import gen_data
from my_logging.loggers import LogMetrics, Tracker
from nn_to_rule_set import fidelity, nn_to_rule_set
from train_model import training_loop
from utilities import acc, set_seed


def train_nn(
    train_dl: DataLoader,
    valid_dl: DataLoader,
    model: nn.Sequential,
    epochs: int,
    l1: float,
) -> tuple[list[float], list[float]]:
    device = "cpu"
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-6)
    tracker = Tracker()
    tracker.add_logger(
        LogMetrics(["timestamp", "epoch", "train_loss", "train_acc"], log_every=20)
    )

    return training_loop(
        model,
        loss_fn,
        optim,
        train_dl,
        valid_dl,
        epochs=epochs,
        lambda1=l1,
        tracker=tracker,
        device=device,
    )


def train_parity(n: int, path: Path, epochs: int, l1: float):
    vars = [f"x{i + 1}" for i in range(n)]
    target_func = PARITY(vars)
    model = nn.Sequential(
        nn.Linear(n, n),
        nn.Tanh(),
        nn.Linear(n, n),
        nn.Tanh(),
        nn.Linear(n, n),
        nn.Tanh(),
        nn.Linear(n, n),
        nn.Tanh(),
        nn.Linear(n, 1),
        nn.Sigmoid(),
        nn.Flatten(0),
    )
    data_path = path / "data.csv"

    # generate a dataset, given a logical function
    data = gen_data(target_func, n=max(1024, int(2**n)))
    # save it in a throwaway folder
    if not os.path.exists(path):
        os.makedirs(path)
    data.to_csv(data_path, index=False, sep=",", mode="w")
    train_dl = DataLoader(FileDataset(data_path), batch_size=64)
    valid_dl = DataLoader(FileDataset(data_path), batch_size=64)
    _ = train_nn(train_dl, valid_dl, model, epochs, l1)
    return model


def main():
    n_vars = 8
    epochs = 2000
    l1 = 5e-5
    name = f"parity_{n_vars}_l{l1}_epoch{epochs}"
    path = Path("runs")
    data_path = path / (name + ".csv")
    model_path = path / (name + ".pth")

    if not os.path.exists(model_path):
        seed = set_seed(0)
        print(f"{seed = }")
        print(f"No pre-trained model found. Starting training...")
        model = train_parity(n_vars, path, epochs=epochs, l1=l1)
        try:
            torch.save(model, model_path)
            print(f"Successfully saved model to {model_path}")
        except Exception as inst:
            print(f"Could not save model to {model_path}: {inst}")

    model = torch.load(model_path)
    print(f"Loading trained model from {model_path}...")
    # get the last updated model
    model = torch.load(model_path)

    print("Transforming model to rule set...")
    rules = nn_to_rule_set(model_path, data_path)
    print(rules)

    # Fidelity: the percentage of test examples for which the classification made by the rules agrees with the neural network counterpart
    # Accuracy: the percentage of test examples that are correctly classified by the rules
    # Consistency: is given if the rules extracted under different training sessions produce the same classifications of test examples
    # Comprehensibility: is determined by measuring the number of rules and the number of antecedents per rule
    train_dl = DataLoader(FileDataset(data_path))
    fid, rules_acc = fidelity(model, rules, data_path)
    nn_acc = acc(model, train_dl, torch.device("cpu"))
    print(f"{nn_acc = }")
    print(f"{fid = }")
    print(f"{rules_acc = }")


if __name__ == "__main__":
    main()
