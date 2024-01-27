import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bool_formula import PARITY
from datasets import FileDataset
from gen_data import gen_data
from models.model_collection import ModelFactory
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
    optim = torch.optim.Adam(model.parameters(), lr=0.007, weight_decay=1e-5)
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


def train_on_data(
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
    # data = gen_data(target_func, shuffle=True)
    data.to_csv(data_path, index=False, sep=",", mode="w")
    train_dl = DataLoader(FileDataset(data_path), batch_size=batch_size)
    valid_dl = DataLoader(FileDataset(data_path), batch_size=batch_size)
    _ = train_nn(train_dl, valid_dl, model, epochs, l1)
    return model


def main():
    seed = 0
    n_vars = 10
    epochs = 4000
    l1 = 0.0
    batch_size = 64
    spec_name = f"parity{n_vars}"
    verbose = False
    name = f"l{l1}_seed{seed}_epoch{epochs}_bs{batch_size}"
    path = Path("runs")
    problem_name = f"parity/{n_vars}"
    data_path = path / problem_name / (name + ".csv")
    model_path = path / problem_name / (name + ".pth")

    if not os.path.isfile(model_path) or not os.path.isfile(data_path):
        seed = set_seed(seed)
        print(f"{seed = }")
        print(f"No pre-trained model found. Starting training...")
        model = ModelFactory.get_model(spec_name, n_vars)
        model = nn.Sequential(
            nn.Linear(n_vars, n_vars),
            nn.Sigmoid(),
            nn.Linear(n_vars, 1),
            nn.Sigmoid(),
            nn.Flatten(0),
        )
        print(model)
        train_on_data(
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
    bg_data = {key: np.array(df[key], dtype=bool) for key in keys}

    print("Transforming model to rule set...")
    ng, q_ng, bg = nn_to_rule_set(model, ng_data, keys, verbose=verbose)

    print("ng = ", ng)
    print("q_ng = ", q_ng)
    print("bg = ", bg)

    nn_out = model(nn_data).detach().numpy()
    ng_out = ng(ng_data)
    q_ng_out = q_ng(ng_data)
    bg_out = bg(bg_data)

    print("----------------- Outputs -----------------")

    print(f"{nn_out = }")
    print(f"{ng_out = }")
    print(f"{q_ng_out = }")
    print(f"{bg_out = }")

    print("----------------- Mean Errors -----------------")

    print("mean error nn - ng: ", np.mean(np.abs(nn_out - ng_out)))
    print("mean error nn - q_ng: ", np.mean(np.abs(nn_out - q_ng_out)))
    print("mean error nn - b_ng: ", np.mean(np.abs(nn_out - bg_out)))

    print("----------------- Fidelity -----------------")

    print("fidelity nn - ng:\t", np.mean(nn_out.round() == ng_out.round()))
    print("fidelity ng - q_ng:\t", np.mean(ng_out.round() == q_ng_out.round()))
    print("fidelity nn - q_ng:\t", np.mean(nn_out.round() == q_ng_out.round()))
    print("fidelity nn - b_ng:\t", np.mean(nn_out.round() == bg_out))

    print("----------------- Final Acc. -----------------")

    print("accuracy nn:\t", np.array(1.0) - np.mean(np.abs(np.round(nn_out) - y)))
    print("accuracy ng:\t", np.array(1.0) - np.mean(np.abs(np.round(ng_out) - y)))
    print("accuracy q_ng:\t", np.array(1.0) - np.mean(np.abs(q_ng_out - y)))
    print("accuracy bg:\t", np.array(1.0) - np.mean(np.abs(bg_out - y)))

    sns.scatterplot()

    # Fidelity: the percentage of test examples for which the classification made by the rules agrees with the neural network counterpart
    # Accuracy: the percentage of test examples that are correctly classified by the rules
    # Consistency: is given if the rules extracted under different training sessions produce the same classifications of test examples
    # Comprehensibility: is determined by measuring the number of rules and the number of antecedents per rule


if __name__ == "__main__":
    main()
