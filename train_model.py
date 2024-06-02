from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.types import Device
from torch.utils.data.dataloader import DataLoader

from my_logging.loggers import Tracker


def train(
    dl: DataLoader,
    model: Module,
    loss_fn: Module,
    optim: Optimizer,
    device: Device,
    tracker=Tracker(),
    lambda1: float = 0,
) -> list[float]:
    """
    Function for the training step of the training loop
    """
    model = model.to(device)
    model.train()
    losses = []

    for X, y in dl:
        tracker.batch_start()
        X = X.to(device)
        y = y.to(device)
        optim.zero_grad()
        y_hat = model(X)
        try:
            loss = loss_fn(y_hat, y.float())
        except:
            temp = 0
        if lambda1 != 0:
            # don't regularize bias
            all_linear1_params = torch.cat(
                [x.view(-1) for x in model.parameters() if x.dim() == 2]
            )
            l1_regularization = lambda1 * torch.norm(all_linear1_params, 1)
            loss = loss + l1_regularization
        losses.append(loss.item())
        loss.backward()
        optim.step()

        tracker.batch_end()

    return losses


@torch.no_grad()
def validate(
    dl: DataLoader,
    model: Module,
    loss_fn: Module,
    device: Device,
) -> list[float]:
    """
    Function for the validation step of the training loop
    """
    model = model.to(device)
    model.eval()
    losses = []

    for X, y_true in dl:
        X = X.to(device)
        y_true = y_true.to(device)
        y_hat = model(X)
        loss = torch.Tensor()
        loss = loss_fn(y_hat, y_true.float())
        losses.append(loss.item())

    return losses


def training_loop(
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    lambda1: float = 0.0,
    device: Device = "cpu",
    tracker: Tracker = Tracker(),
    scheduler: Optional[MultiStepLR] = None,
) -> pd.DataFrame:
    tracker.training_start(model, train_loader, valid_loader, criterion)
    # Train model
    while not tracker.stop_condition():
        tracker.epoch_start()
        # training
        train_loss = train(
            train_loader, model, criterion, optimizer, device, tracker, lambda1
        )

        # validation
        with torch.no_grad():
            valid_loss = validate(valid_loader, model, criterion, device)

        if scheduler is not None:
            scheduler.step()

        tracker.epoch_end(train_loss, valid_loss)
    return tracker.training_end()
