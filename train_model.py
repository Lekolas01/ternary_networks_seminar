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
    assert not torch.any(model[0].weight.isnan())

    for X, y in dl:
        assert not torch.any(model[0].weight.isnan())
        tracker.batch_start()
        X = X.to(device)
        y = y.to(device)
        y_hat = model(X)
        optim.zero_grad()
        assert not torch.any(model[0].weight.isnan())
        try:
            assert not torch.any(model[0].weight.isnan())
            loss = loss_fn(y_hat, y.float())
            assert not torch.any(model[0].weight.isnan())
        except:
            temp = 0
        if lambda1 != 0:
            assert not torch.any(model[0].weight.isnan())
            # don't regularize bias
            all_linear1_params = torch.cat(
                [x.view(-1) for x in model.parameters() if x.dim() == 2]
            )
            l1_regularization = lambda1 * torch.norm(all_linear1_params, 1)
            loss = loss + l1_regularization
            assert not torch.any(model[0].weight.isnan())
        losses.append(loss.item())
        assert not torch.any(model[0].weight.isnan())
        try:
            assert not torch.any(model[0].weight.isnan())
            loss.backward()
            assert not torch.any(model[0].weight.isnan())
        except:
            print("hi")
            temp = 0
        assert not torch.any(model[0].weight.isnan())
        optim.step()
        assert not torch.any(model[0].weight.isnan())
        tracker.batch_end()
        assert not torch.any(model[0].weight.isnan())
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
        assert not torch.any(model[0].weight.isnan())
        tracker.epoch_start()
        # training
        assert not torch.any(model[0].weight.isnan())
        train_loss = train(
            train_loader, model, criterion, optimizer, device, tracker, lambda1
        )
        assert not torch.any(model[0].weight.isnan())

        # validation
        with torch.no_grad():
            valid_loss = validate(valid_loader, model, criterion, device)
        assert not torch.any(model[0].weight.isnan())

        if scheduler is not None:
            scheduler.step()
        assert not torch.any(model[0].weight.isnan())

        tracker.epoch_end(train_loss, valid_loss)
        assert not torch.any(model[0].weight.isnan())
    return tracker.training_end()
