import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader


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


def distance_from_int_precision(x: np.ndarray):
    """
    Calculates the avg. distance of x w.r.t. the nearest integer. 
    x must be in range [-1, 1]
    When doing many update steps with WDR Regularizer, this distance should decrease.
    """
    assert(np.allclose(x, np.zeros_like(x), atol = 1, rtol=0))
    ans = np.full_like(x, 2)
    bases = [-1, 0, 1]
    for base in bases:
        d = np.abs(x - base)
        ans = np.where(d < ans, d, ans)
    minus_ones = np.count_nonzero(x <= -0.5) / x.shape[0]
    plus_ones = np.count_nonzero(x >= 0.5) / x.shape[0]
    zeros = 1 - minus_ones - plus_ones
    return ans, zeros


def get_all_weights(model: nn.Module):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)
    return params
