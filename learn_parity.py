import os
import sys

import torch

from bool_formula import Activation
from generate_parity_dataset import parity_df
from models.model_collection import ModelFactory
from train_mlp import train_mlp
from utilities import acc

# Grid Search over NN training hyperparameters
#   call train_mlp.py
# For each trained model, extract rule set
# save all rule sets
# create graph for rule sets: dimensions are complexity and accuracy


def main(k: int):
    # Generate dataframe for parity5
    os.mkdir(f"runs/parity{k}")
    df = parity_df(k=k, shuffle=True, n=1000)
    df.to_csv(f"runs/parity{k}/data.csv")
    print(df.head())
    print(f"{df.shape = }")

    # Do a single NN training run on this dataset
    seed = 0
    bs = 32
    epochs = 6000
    l1 = 0.0
    wd = 0.0

    learning_rates = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    batch_sizes = [32, 64]
    for lr in learning_rates:
        for bs in batch_sizes:
            spec = [
                (k, k, Activation.TANH),
                (k, k, Activation.TANH),
                (k, k, Activation.TANH),
                (k, 1, Activation.SIGMOID),
            ]
            model = ModelFactory.get_model_by_name(f"parity{k}")
            losses, dl = train_mlp(df, model, seed, bs, lr, epochs, l1, wd)
            accuracy = acc(model, dl, "cpu")
            torch.save(model, f"runs/parity{k}/lr{lr}_bs{bs}_acc{accuracy}.pth")


if __name__ == "__main__":
    main(int(sys.argv[1]))
