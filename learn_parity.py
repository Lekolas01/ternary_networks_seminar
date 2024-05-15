import contextlib
import itertools
import os
import shutil
import sys
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import torch
from genericpath import isfile

from bool_formula import Activation
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
    seed = 0
    set_seed(seed)
    # Generate dataframe for parity
    f_root = f"runs/parity{k}"
    f_data = f"{f_root}/data.csv"
    f_models = f"{f_root}/models"
    f_runs = f"{f_root}/runs.csv"
    f_losses = f"{f_root}/losses.csv"
    if os.path.exists(f_root):
        shutil.rmtree(f"{f_root}")

    os.makedirs(f_models, exist_ok=True)
    df = parity_df(k=k, shuffle=True, n=1000)

    # write dataset to data.csv
    df.to_csv(f_data, index=False)

    # Do a single NN training run on this dataset
    epochs = 1500
    l1 = 0.0
    bs = 32
    wd = 0.0

    lrs = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
    n_layers = [1, 2, 3]
    runs = pd.DataFrame(
        columns=["idx", "lr", "n_layer", "seed", "epochs", "bs", "l1", "wd"]
    )
    for idx, (lr, n_layer) in enumerate(itertools.product(lrs, n_layers)):
        model_name = f"runs/parity{k}/models/{idx}.pth"
        spec: NNSpec = [(k, k, Activation.TANH) for _ in range(n_layer)]
        spec.append(((k, 1, Activation.SIGMOID)))

        model = ModelFactory.get_model_by_spec(spec)
        metrics, dl = train_mlp(df, model, seed, bs, lr, epochs, l1, wd)

        # add run to runs file
        cols = [idx, lr, n_layer, seed, epochs, bs, l1, wd]
        runs.loc[idx] = [str(val) for val in cols]
        runs.to_csv(f_runs, mode="w", header=True, index=False)

        # save trained model
        torch.save(model, model_name)

        # append losses to losses.csv
        metrics.insert(loc=0, column="idx", value=idx)
        metrics.insert(loc=1, column="epoch", value=[i + 1 for i in range(epochs)])
        metrics.to_csv(f_losses, mode="a", index=False, header=(idx == 0))

        keys = list(df.columns)
        keys.pop()
        y = np.array(df["target"])
        ng_data = {key: np.array(df[key], dtype=float) for key in keys}
        ng, q_ng, bg = nn_to_rule_set(model, ng_data, keys)
        print(f"{bg.complexity() = }")

    # TODO für morgen:
    #   Regeln lernen und miteinander vergleichen können


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
    main(args.k)
