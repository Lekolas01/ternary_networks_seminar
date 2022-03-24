import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from config import Configuration

from tracking import Tracker


def train(train_loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: Optimizer, device: str, dry_run=False, **kwargs) -> float:
    '''
    Function for the training step of the training loop
    '''
    model.train()
    running_loss = 0

    for X, y in train_loader:
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
    
        y_hat, probs = model(X)
        loss = criterion(y_hat, y)
        if (hasattr(model, 'regularization')):
            loss = loss + model.regularization()
        
        running_loss += loss.item() * X.size(0)

        loss.backward()
        optimizer.step()
        if dry_run: 
            epoch_loss = running_loss / len(train_loader.dataset)
            return epoch_loss
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


@torch.no_grad()
def validate(valid_loader: DataLoader, model: nn.Module, criterion: nn.Module, device: str, dry_run=False, **kwargs) -> float:
    '''
    Function for the validation step of the training loop
    '''
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)

        y_hat, probs = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

        if dry_run: 
            epoch_loss = running_loss / len(valid_loader.dataset)
            return epoch_loss

    epoch_loss = running_loss / len(valid_loader.dataset)
    return epoch_loss


def training_loop(model: nn.Module, criterion: nn.Module, optimizer: Optimizer, train_loader: DataLoader, \
    valid_loader: DataLoader, epochs: int, device: str, tracker:Tracker=None, scheduler: MultiStepLR=None, **kwargs):
    
    tracker.loop_init(model, train_loader, valid_loader, **kwargs)
    # Train model
    for epoch in range(epochs):
        # training
        train_loss = train(train_loader, model, criterion, optimizer, device, **kwargs)

        # validation
        with torch.no_grad():
            valid_loss = validate(valid_loader, model, criterion, device, **kwargs)

        tracker.track_loss(train_loss, valid_loss)
        if scheduler is not None:
            scheduler.step()
    
    train_losses, valid_losses = tracker.summarise()
    return (train_losses, valid_losses)

