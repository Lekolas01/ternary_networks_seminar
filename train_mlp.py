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
    data_name = args.data
    model_name = args.model

    data_path = Path("data/generated") / f"{data_name}.csv"
    problem_path = Path(f"runs/{data_name}")
    model_path = problem_path / f"{model_name}.pth"

    if not os.path.isdir(problem_path):
        print(f"Creating new directory at {problem_path}...")
        os.mkdir(problem_path)

    if args.new or not os.path.isfile(model_path):
        seed = set_seed(args.seed)
        print(f"{seed = }")
        print(f"No pre-trained model found. Starting training...")
        model = ModelFactory.get_model(model_name)

        train_dl = DataLoader(
            FileDataset(data_path), batch_size=args.batch_size, shuffle=True
        )
        valid_dl = DataLoader(
            FileDataset(data_path), batch_size=args.batch_size, shuffle=True
        )
        loss_fn = nn.BCELoss()
        optim = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.wd,
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
            epochs=args.epochs,
            lambda1=args.l1,
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
