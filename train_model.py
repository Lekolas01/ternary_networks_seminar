from typing import Optional
import torch
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from loggers.loggers import Tracker
from torch.types import Device


def train(
    dl: DataLoader,
    model: Module,
    loss_fn: Module,
    optim: Optimizer,
    device: Device,
    tracker=Tracker(),
) -> list[float]:
    """
    Function for the training step of the training loop
    """
    model.train()
    losses = []

    for X, y in dl:
        tracker.batch_start()
        X = X.to(device)
        y = y.to(device)
        optim.zero_grad()
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        losses.append(loss.item())
        loss.backward()
        optim.step()
        tracker.batch_end()

    return losses


@torch.no_grad()
def validate(
    dl: DataLoader, model: Module, loss_fn: Module, device: Device
) -> list[float]:
    """
    Function for the validation step of the training loop
    """
    model.eval()
    losses = []

    for X, y_true in dl:
        X = X.to(device)
        y_true = y_true.to(device)
        y_hat = model(X)
        loss = loss_fn(y_hat, y_true)
        losses.append(loss.item())

    return losses


def training_loop(
    model: Module,
    criterion: Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    device: Device = "cpu",
    tracker: Tracker = Tracker(),
    scheduler: Optional[MultiStepLR] = None,
) -> tuple[list[float], list[float]]:
    tracker.training_start(model, train_loader, valid_loader, criterion)
    # Train model
    for epoch in range(epochs):
        tracker.epoch_start()
        # training
        train_loss = train(train_loader, model, criterion, optimizer, device, tracker)
        print()

        # validation
        with torch.no_grad():
            valid_loss = validate(valid_loader, model, criterion, device)

        if scheduler is not None:
            scheduler.step()

        tracker.epoch_end(train_loss, valid_loss)

    return tracker.training_end()