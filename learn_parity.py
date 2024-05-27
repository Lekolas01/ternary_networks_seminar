import contextlib
import itertools
import os
import shutil
import sys
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from genericpath import isfile

from bool_formula import Activation
from bool_parse import ExpressionEvaluator
from gen_data import gen_data
from generate_parity_dataset import parity_df
from models.model_collection import ModelFactory, NNSpec
from rule_extraction import nn_to_rule_set
from train_mlp import train_mlp
from utilities import acc, set_seed

# Grid Search over NN training hyperparameters
#   call train_mlp.py
# For each trained model, extract rule set
# save all rule sets
# create graph for rule sets: dimensions are complexity and accuracy


def main(k: int):
    seed = 1
    set_seed(seed)
    # Generate dataframe for parity
    f_root = f"runs/parity{k}"
    f_data = f"{f_root}/data.csv"
    f_models = f"{f_root}/models"
    f_runs = f"{f_root}/runs.csv"
    f_losses = f"{f_root}/losses.csv"

    os.makedirs(f_models, exist_ok=True)
    df = parity_df(k=k, shuffle=False, n=1024)
    # write dataset to data.csv
    df.to_csv(f_data, index=False)
    df = pd.read_csv(f_data)
    print(f"Generated dataset with shape {df.shape}")

    # Do a single NN training run on this dataset
    max_epochs = 6000
    bs = 64
    wd = 0.0

    lrs = [1e-3, 3e-3, 1e-2, 3e-2]
    l1s = [0.0, 1e-5, 1e-4]
    n_layers = [1, 2, 3]
    runs = pd.DataFrame(
        columns=["idx", "lr", "n_layer", "seed", "epochs", "bs", "l1", "wd"]
    )
    n_runs = len(l1s) * len(lrs) * len(n_layers)
    for idx, (l1, lr, n_layer) in enumerate(itertools.product(l1s, lrs, n_layers)):
        print(f"{max_epochs = }\t|{bs = }\t|{wd = }\t|{lr = }\t|{n_layer = }\t|{l1 = }")
        model_path = f"{f_models}/{idx}.pth"
        if os.path.isfile(model_path):
            print("Model already trained. Skipping this training session...")
            continue
        spec: NNSpec = [(k, k, Activation.TANH) for _ in range(n_layer)]
        spec.append(((k, 1, Activation.SIGMOID)))

        model = ModelFactory.get_model_by_spec(spec)

        metrics, dl = train_mlp(df, model, seed, bs, lr, max_epochs, l1, wd)

        # add run to runs file
        cols = [idx, lr, n_layer, seed, max_epochs, bs, l1, wd]
        runs.loc[idx] = [str(val) for val in cols]
        runs.to_csv(f_runs, mode="w", header=True, index=False)
        n_epochs = len(metrics["train_loss"])

        # save trained model
        torch.save(model, model_path)

        # append losses to losses.csv
        metrics.insert(loc=0, column="idx", value=idx)
        metrics.insert(loc=1, column="epoch", value=[i + 1 for i in range(n_epochs)])
        metrics.to_csv(f_losses, mode="a", index=False, header=(idx == 0))

    complexities = []
    accs = []
    for idx in range(n_runs):
        model_path = f"{f_models}/{idx}.pth"
        model = torch.load(model_path)
        keys = list(df.columns)
        keys.pop()
        y = np.array(df["target"])
        ng_data = {key: np.array(df[key], dtype=float) for key in keys}
        _, _, bg = nn_to_rule_set(model, ng_data, keys)
        bg_data = {key: np.array(data, dtype=bool) for key, data in ng_data.items()}
        pred = bg(bg_data)
        complexities.append(bg.complexity())
        accs.append(1 - sum(abs(pred - y)) / len(pred))

    # TODO für morgen:
    #   Regeln lernen und miteinander vergleichen können
    sns.scatterplot(x=complexities, y=accs)
    plt.show()


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Train an MLP on a binary classification task with an ADAM optimizer."
    )
    parser.add_argument("k", help="Arity")
    parser.add_argument(
        "--new",
        action="store_true",
        help="If specified, the model will be retrained.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(int(args.k))
