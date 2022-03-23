import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

from tracking import *


def get_accuracy(model: nn.Module, data_loader: DataLoader, device: str) -> float:
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    correct_pred = 0
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            if y_prob.dim() == 1:
                predicted_labels = torch.round(y_prob)
            else:
                _, predicted_labels = torch.max(y_prob, 1)
            correct_pred += (predicted_labels == y_true).sum()
            n += y_true.size(0)

    return correct_pred.float() / n


def train(train_loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: Optimizer, device: str) -> float:
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
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


@torch.no_grad()
def validate(valid_loader: DataLoader, model: nn.Module, criterion: nn.Module, device: str) -> float:
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

    epoch_loss = running_loss / len(valid_loader.dataset)
    return epoch_loss


def training_loop(model: nn.Module, criterion: nn.Module, optimizer: Optimizer, train_loader: DataLoader, \
    valid_loader: DataLoader, epochs: int, device: str, tracker:Tracker=None, scheduler: MultiStepLR=None, **tracking_args):
    
    tracker.loop_init(model, **tracking_args)
    # Train model
    for epoch in range(epochs):
        # training
        train_loss = train(train_loader, model, criterion, optimizer, device)

        # validation
        with torch.no_grad():
            valid_loss = validate(valid_loader, model, criterion, device)
            train_acc = get_accuracy(model, train_loader, device)
            valid_acc = get_accuracy(model, valid_loader, device)

        tracker.track_loss(train_loss, valid_loss, train_acc, valid_acc)
        if scheduler is not None:
            scheduler.step()
    
    train_losses, valid_losses = tracker.summarise()
    return (train_losses, valid_losses)

