import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import FileDataset
from models.model_collection import ModelFactory
from my_logging.loggers import LogMetrics, Tracker
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
        "data",
        help="The name of the dataset (the file must exist in the data/generated folder).",
    )
    parser.add_argument(
        "model",
        help="The name of the neural net configuration - see models.model_collection.py.",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="If specified, the model will be retrained, even if it is already saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=seed,
        help="The seed for the training configuration. Running the script with the same seed will result in the same output.",
    )
    parser.add_argument(
        "--l1",
        type=float,
        default=l1,
        help="L1 loss that gets added to the training procedure.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=epochs,
        help="The number of training epochs on which to train the neural net.",
    )
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=lr, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=weight_decay, help="Weight decay")
    return parser.parse_args()


def main():
    args = get_arguments()
    return train_mlp(
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


def train_mlp(
    data_name: str,
    model_name: str,
    new: bool,
    seed: int,
    batch_size: int,
    lr: float,
    epochs: int,
    l1: float,
    wd: float,
):
    data_path = Path("data/generated") / f"{data_name}.csv"
    problem_path = Path(f"runs/{data_name}")
    model_path = problem_path / f"{model_name}.pth"

    if not os.path.isdir(problem_path):
        print(f"Creating new directory at {problem_path}...")
        os.mkdir(problem_path)

    if not new and os.path.isfile(model_path):
        print("Model has been trained already. Training procedure cancelled.")
        return
    seed = set_seed(seed)
    print(f"{seed = }")
    print(f"No pre-trained model found. Starting training...")
    model = ModelFactory.get_model(model_name)

    train_dl = DataLoader(FileDataset(data_path), batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(FileDataset(data_path), batch_size=batch_size, shuffle=True)
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=wd,
    )
    tracker = Tracker()
    tracker.add_logger(
        LogMetrics(["timestamp", "epoch", "train_loss", "train_acc"], log_every=50)
    )

    losses = training_loop(
        model,
        loss_fn,
        optim,
        train_dl,
        valid_dl,
        epochs=epochs,
        lambda1=l1,
        tracker=tracker,
        device="cpu",
    )

    try:
        torch.save(model, model_path)
        print(f"Successfully saved model to {model_path}")
    except Exception as inst:
        print(f"Could not save model to {model_path}: {inst}")


if __name__ == "__main__":
    main()
