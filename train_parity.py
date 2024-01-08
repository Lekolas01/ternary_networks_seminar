import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bool_formula import PARITY
from datasets import FileDataset
from gen_data import gen_data
from my_logging.loggers import LogMetrics, Tracker
from neuron import nn_to_rule_set
from train_model import training_loop
from utilities import set_seed


def train_nn(
    train_dl: DataLoader,
    valid_dl: DataLoader,
    model: nn.Sequential,
    epochs: int,
    l1: float,
) -> tuple[list[float], list[float]]:
    device = "cpu"
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
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


def train_on_parity(
    model: nn.Sequential,
    n: int,
    data_path: Path,
    epochs: int,
    l1: float,
    batch_size: int,
):
    vars = [f"x{i + 1}" for i in range(n)]
    target_func = PARITY(vars)

    # generate a dataset, given a logical function
    data = gen_data(target_func, n=max(1024, int(2**n)))
    data.to_csv(data_path, index=False, sep=",", mode="w")
    train_dl = DataLoader(FileDataset(data_path), batch_size=batch_size)
    valid_dl = DataLoader(FileDataset(data_path), batch_size=batch_size)
    _ = train_nn(train_dl, valid_dl, model, epochs, l1)
    return model


def main():
    seed = 1
    n_vars = 10
    epochs = 3000
    l1 = 2e-5
    batch_size = 64
    name = f"parity_{n_vars}_l{l1}_epoch{epochs}_bs{batch_size}"
    path = Path("runs")
    data_path = path / (name + ".csv")
    model_path = path / (name + ".pth")

    if not os.path.isfile(model_path) or not os.path.isfile(data_path):
        seed = set_seed(seed)
        print(f"{seed = }")
        print(f"No pre-trained model found. Starting training...")
        model = nn.Sequential(
            nn.Linear(n_vars, n_vars),
            nn.Tanh(),
            nn.Linear(n_vars, 1),
            nn.Sigmoid(),
            nn.Flatten(0),
        )
        train_on_parity(
            model, n_vars, data_path, epochs=epochs, l1=l1, batch_size=batch_size
        )
        try:
            torch.save(model, model_path)
            print(f"Successfully saved model to {model_path}")
        except Exception as inst:
            print(f"Could not save model to {model_path}: {inst}")

    print(f"Loading trained model from {model_path}...")
    # get the last updated model
    model = torch.load(model_path)

    df = pd.read_csv(data_path, dtype=float)
    keys = list(df.columns)
    keys.pop()  # remove target column
    y = df["target"]
    ng_data = {key: np.array(df[key], dtype=float) for key in keys}
    nn_data = np.stack([ng_data[key] for key in keys], axis=1)
    nn_data = torch.Tensor(nn_data)

    print("Transforming model to rule set...")
    ng, q_ng, b_ng = nn_to_rule_set(model, ng_data, keys)

    print("ng = ", ng)
    print("q_ng = ", q_ng)
    # print("b_ng = ", b_ng)

    nn_out = model(nn_data).detach().numpy().round()
    ng_out = ng(ng_data).round()
    q_ng_out = q_ng(ng_data)
    # b_ng_out = b_ng(ng_data)

    print(f"{nn_out = }")
    print(f"{ng_out = }")
    print(f"{q_ng_out = }")
    # print(f"{b_ng_out = }")

    print("--------------------------------------")
    print("mean error nn - ng: ", np.mean(np.abs(nn_out - ng_out)))
    print("mean error nn - q_ng: ", np.mean(np.abs(nn_out - q_ng_out)))
    # print("mean error nn - b_ng: ", np.mean(np.abs(nn_out - b_ng_out)))

    print("fidelity nn - ng:\t", np.mean(nn_out == ng_out))
    print("fidelity nn - q_ng:\t", np.mean(nn_out == q_ng_out))
    # print("fidelity nn - b_ng:\t", np.mean(nn_out == b_ng_out))

    print("accuracy nn:\t", np.array(1.0) - np.mean(np.abs(np.round(nn_out) - y)))
    print("accuracy ng:\t", np.array(1.0) - np.mean(np.abs(ng_out - y)))
    print("accuracy q_ng:\t", np.array(1.0) - np.mean(np.abs(q_ng_out - y)))
    print("--------------------------------------")
    exit()

    # Fidelity: the percentage of test examples for which the classification made by the rules agrees with the neural network counterpart
    # Accuracy: the percentage of test examples that are correctly classified by the rules
    # Consistency: is given if the rules extracted under different training sessions produce the same classifications of test examples
    # Comprehensibility: is determined by measuring the number of rules and the number of antecedents per rule


if __name__ == "__main__":
    main()
