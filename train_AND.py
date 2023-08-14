import pandas as pd
from pathlib import Path
import torch
from models.logical_AND import ANDNet
from dataloading import FileDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from train_model import training_loop
from loggers.loggers import *
import torch.utils.data.sampler

torch.random.manual_seed(1)
model = ANDNet(2)
data_path = Path("data", "generated", "logical_AND", "data.csv")
data = pd.read_csv(data_path)
dataset = FileDataset(str(data_path))

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=False)

print(model)

loss_fn = nn.BCELoss()
optim = torch.optim.SGD(model.parameters(), 1)
t = Tracker()
t.add_logger(LogMetrics(["timestamp", "epoch", "train_loss", "train_acc", "valid_acc"]))
#t.add_logger(LogModel())
losses = training_loop(model, loss_fn, optim, dataloader, dataloader, 50, "cpu", t)

print(len(dataset))
a = torch.tensor([4]).squeeze()
print(a)
print(a.shape)