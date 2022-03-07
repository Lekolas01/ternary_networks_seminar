import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

import pandas as pd
import os, sys

class AdultDataset(torch.utils.data.Dataset):
    """
    A pytorch Dataset wrapper around the adult dataset: https://archive.ics.uci.edu/ml/datasets/adult
    It assumes that the files have not been changed after download.
    """
    # all headers
    NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target']
    # all features that have an order on its data defined, i.e. that can be label encoded instead of one-hot encoded
    ORDINALS = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    def __init__(self, path):
        # TODO: ignore incomplete data
         
        self.adult_frame = pd.read_csv(path, names = self.NAMES)
        #self.adult_frame.columns

    def __len__(self):
        return len(self.adult_frame)

    def __getitem__(self, idx):
        return None


def get_mnist_dataloader(train, samples: int=None, **dl_args):
    transform = transforms.Compose([transforms.Resize((32, 32)),
        transforms.ToTensor()])

    dataset = datasets.MNIST(root='data/', 
        train=train, 
        transform=transform,
        download=True)

    if (samples is not None and 1 <= samples < len(dataset)):
        dataset = torch.utils.data.Subset(dataset, torch.arange(samples))

    data_loader = DataLoader(dataset, **dl_args)

    return data_loader


if __name__ == '__main__':
    path = os.path.join('data', 'adult', 'adult.data')
    ds = AdultDataset(path)

    print(len(ds))
    print(ds)

    print(sys.getsizeof(ds))
    print(sys.getsizeof(ds.adult_frame))
    print(ds.adult_frame.columns)
