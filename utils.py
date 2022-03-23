import numpy as np
import torch
import torch.nn as nn


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
    return ans, (minus_ones, zeros, plus_ones)


def get_all_weights(model: nn.Module):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)
    return params
