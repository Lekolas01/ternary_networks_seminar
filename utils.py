import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.types import Device

def accuracy(model: nn.Module, data_loader: DataLoader, device: Device) -> float:
    """
    Compute the accuracy of a model on the data held by a data loader.
    """
    correct_pred = 0.0
    n = 0
    if next(model.parameters()).device != device:
        model = model.to(device)

    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:
            X = X.to(device)
            y_true = y_true.to(device)

            y_hat = model(X)
            if y_hat.dim() == 1:
                predicted_labels = torch.round(y_hat)
            else:
                _, predicted_labels = torch.max(y_hat, 1)
            correct_pred += (predicted_labels == y_true).sum()
            n += y_true.size(0)
    return float(correct_pred) / n


def distance_from_int_precision(x: np.ndarray):
    """
    Calculates the avg. distance of x w.r.t. the nearest integer.
    x must be in range [-1, 1]
    When doing many update steps with WDR Regularizer, this distance should decrease.
    """
    assert np.allclose(x, np.zeros_like(x), atol=1, rtol=0)
    ans = np.full_like(x, 2)
    bases = [-1, 0, 1]
    for base in bases:
        d = np.abs(x - base)
        ans = np.where(d < ans, d, ans)
    minus_ones = np.count_nonzero(x <= -0.5) / x.shape[0]
    plus_ones = np.count_nonzero(x >= 0.5) / x.shape[0]
    zeros = 1 - minus_ones - plus_ones
    return np.mean(ans), zeros


def get_all_weights(model: nn.Module):
    params = [param.view(-1) for param in model.parameters()]
    params = torch.cat(params)
    return params

