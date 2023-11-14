import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data.sampler
from torch.utils.data import DataLoader
from adult import Adult

from datasets import get_datasets
from my_logging.loggers import LogMetrics, Tracker
from train_model import training_loop


# ------------------- my dataset --------------
if __name__ == "__main__":
    """
    model = nn.Sequential(
        nn.Linear(104, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.Flatten(0),
    )
    """
    model = nn.Sequential(
        nn.Linear(104, 80),
        nn.ReLU(),
        nn.Linear(80, 1),
        nn.Flatten(0),
    )

    layer_1 = model[0]
    assert isinstance(layer_1, nn.Linear)
    _, shape_in = layer_1.weight.shape

    # ------------- their dataset

    train_set = Adult(root="datasets", download=True)
    test_set = Adult(root="datasets", train=False, download=True)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

    # train a neural network on the dataset
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    tracker = Tracker()
    tracker.add_logger(
        LogMetrics(
            ["timestamp", "epoch", "train_loss", "train_acc", "valid_acc"], log_every=1
        )
    )
    train_losses, test_losses = training_loop(
        model,
        loss_fn,
        optim,
        train_loader,
        test_loader,
        epochs=400,
        tracker=tracker,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    x = np.arange(len(train_losses))
    model.train(False)
    plt.plot(x, train_losses, label="Train Loss")
    plt.plot(x, test_losses, label="Validation Loss")
    plt.show()
