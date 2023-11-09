import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data.sampler

from dataloading import DataloaderFactory
from my_logging.loggers import LogMetrics, Tracker
from train_model import training_loop

if __name__ == "__main__":
    # model = nn.Sequential(
    #    nn.Linear(104, 40),
    #    nn.Tanh(),
    #    nn.Linear(40, 10),
    #    nn.Tanh(),
    #    nn.Linear(10, 1),
    #    nn.Flatten(0),
    # )

    model = nn.Sequential(
        nn.Linear(104, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(512, 1),
        nn.Flatten(0),
    )

    layer_1 = model[0]
    assert isinstance(layer_1, nn.Linear)
    _, shape_in = layer_1.weight.shape

    train_loader, valid_loader = DataloaderFactory(
        ds="adult", shuffle=True, batch_size=64
    )

    # train a neural network on the dataset
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    tracker = Tracker()
    tracker.add_logger(
        LogMetrics(["timestamp", "epoch", "train_loss", "train_acc"], log_every=10)
    )
    train_losses, valid_losses = training_loop(
        model,
        loss_fn,
        optim,
        train_loader,
        valid_loader,
        epochs=100,
        tracker=tracker,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    x = np.arange(len(train_losses))
    model.train(False)
    plt.plot(x, train_losses, label="Train Loss")
    plt.plot(x, valid_losses, label="Validation Loss")
    plt.show()
