import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(train, samples: int=None, **dl_args):
    transform = transforms.Compose([transforms.Resize((32, 32)),
        transforms.ToTensor()])

    dataset = datasets.MNIST(root='data/', 
        train=train, 
        transform=transform,
        download=True)

    if (samples is not None and 1 <= samples < len(dataset)):
        dataset = data_utils.Subset(dataset, torch.arange(samples))

    data_loader = DataLoader(dataset, **dl_args)

    return data_loader
    

def distance_from_int_precision(x: np.ndarray):
    """
    Calculates the avg. distance of x w.r.t. the nearest integer. 
    x must be in range [-1, 1]
    When doing many update steps with WDR Regularizer, this distance should decrease.
    """
    #assert(torch.allclose(x, torch.zeros_like(x), atol = 1, rtol=0))
    ans = np.full_like(x, 2)
    bases = [-1, 0, 1]
    for base in bases:
        d = np.abs(x - base)
        ans = np.where(d < ans, d, ans)
        
    return ans


def get_all_weights(model: nn.Module):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    params = torch.cat(params)
    return params

