from argparse import ArgumentParser, Namespace

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import FileDataset
from my_logging.loggers import LogMetrics, SaveModel, Tracker
from train_model import training_loop
from utilities import set_seed


def get_arguments() -> Namespace:
    seed = 1
    epochs = 1000
    batch_size = 64
    lr = 0.006
    weight_decay = 0.0
    l1 = 0.0

    parser = ArgumentParser(
        description="Train an MLP on a binary classification task with an ADAM optimizer."
    )
    parser.add_argument(
        "data", help="The name of the dataset (starting from project root)."
    )
    parser.add_argument(
        "model", help="Name of neural net configuration - see model_collection.py."
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="If specified, the model will always be retrained.",
    )
    parser.add_argument("--seed", type=int, default=seed, help="Seed for NN training.")
    parser.add_argument("--l1", type=float, default=l1, help="L1 loss lambda value.")
    parser.add_argument(
        "--epochs", type=int, default=epochs, help="The number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=lr, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=weight_decay, help="Weight decay")
    return parser.parse_args()


def main():
    args = get_arguments()
    model, losses = train_mlp(
        args.data,
        args.model,
        args.new,
        args.seed,
        args.batch_size,
        args.lr,
        args.epochs,
        args.l1,
        args.wd,
    )

    try:
        torch.save(model, model_path)
        print(f"Successfully saved model to {model_path}")
    except Exception as inst:
        print(f"Could not save model to {model_path}: {inst}")


def train_mlp(
    data: pd.DataFrame,
    model: nn.Sequential,
    seed: int,
    batch_size: int,
    lr: float,
    epochs: int,
    l1: float,
    wd: float,
):
    seed = set_seed(seed)

    train_dl = DataLoader(FileDataset(data), batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(FileDataset(data), batch_size=batch_size, shuffle=True)
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=wd,
    )
    tracker = Tracker(epochs=epochs)
    tracker.add_logger(
        LogMetrics(
            ["timestamp", "epoch", "train_loss", "train_acc"],
        )
    )

    return (
        training_loop(
            model,
            loss_fn,
            optim,
            train_dl,
            valid_dl,
            epochs=epochs,
            lambda1=l1,
            tracker=tracker,
            device="cpu",
        ),
        train_dl,
    )


if __name__ == "__main__":
    main()
