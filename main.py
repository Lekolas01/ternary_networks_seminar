from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Configuration, Grid, read_grid
from datasets import get_dataset
from models.model_collection import ModelFactory
from my_logging.checkpoints import Checkpoints
from my_logging.loggers import Plotter, Tracker
from train_model import training_loop


def run(
    conf: Configuration, epochs: int, tracker=Tracker(), **kwargs
) -> tuple[list[float], list[float]]:
    torch.manual_seed(conf.seed)

    # check device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ModelFactory(
        str(conf.data), bool(conf.ternary), float(str(conf.a)), float(str(conf.b))
    ).to(device)

    train_ds, valid_ds = get_dataset(str(conf.data))
    assert isinstance(conf.batch_size, int)
    train_loader = DataLoader(train_ds, batch_size=int(conf.batch_size), shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=int(conf.batch_size), shuffle=True)

    criterion = nn.CrossEntropyLoss() if conf.data == "mnist" else nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(str(conf.lr)))
    scheduler = None
    if conf.schedule_lr:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=[80, 120], gamma=0.1
        )

    return training_loop(
        model,
        criterion,
        optimizer,
        train_loader,
        valid_loader,
        epochs,
        0,
        device,
        tracker,
        scheduler,
        **kwargs
    )


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Generic Training Procedure for a NN with hyperparameters specified by a configuration."
    )
    parser.add_argument("config", help="The name of the training configuration.")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=5,
        help="For how many epochs you want to train.",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="If specified, will save the trained model in a dedicated folder (named after config).",
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="If true, will plot the current loss at each iteration.",
    )
    args = parser.parse_args()
    return args


config = "logical_AND"
epochs = 5
save = False
plot = False


def run_grid(grid: Grid, epochs: int, tracker: Tracker):
    for idx, conf in enumerate(grid):
        print(idx, ":", conf)
        train_err, _ = run(conf, epochs, tracker=tracker)


if __name__ == "__main__":
    conf_path = "configs.json"
    grid = read_grid(conf_path, config)

    tracker = Tracker()
    # tracker.add_logger(Progress(log_every=1)) # What is Progress?
    if save:
        save_path = Path("runs") / config
        tracker.add_logger(Checkpoints(path=save_path, model_every=epochs))
    if plot:
        tracker.add_logger(Plotter())

    run_grid(grid, epochs, tracker)
