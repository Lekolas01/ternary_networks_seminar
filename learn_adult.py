import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data.sampler
from torch.utils.data import DataLoader
from ucimlrepo import fetch_ucirepo

from datasets import FileDataset, get_dataset
from my_logging.loggers import LogMetrics, Tracker
from train_model import training_loop


def main():
    # fetch dataset
    adult = fetch_ucirepo(id=2)

    # data (as pandas dataframes)
    X = adult.data.features
    y = adult.data.targets
    target_col = y.columns[0]
    print(target_col)
    print(f"{type(adult) = }")
    print(f"{type(X) = }")
    print(f"{type(y) = }")
    print(f"{X.shape = }")
    print(f"{y.shape = }")
    temp = X.join(y)
    print(f"{temp.shape = }")
    print(f"{temp.columns = }")
    print(f"{X.isnull().values.any()}")
    print(f"{y.isnull().values.any()}")
    print(f"{X.isnull().any()}")

    # metadata
    print(adult.metadata)

    # variable information
    print(adult.variables)
    ds = FileDataset(temp, (0, 2 / 3), target_col)
    dl = DataLoader(ds, batch_size=32, shuffle=True)


if __name__ == "__main__":
    main()
