from argparse import *

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data.dataloader import DataLoader
from torch.optim import Optimizer

from config import Grid, Configuration, read_grid
from tracking import *
from models.lenet5 import get_model
import dataloading

# ## Helper Functions

def get_accuracy(model: nn.Module, data_loader: DataLoader, device: str):
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
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def R(weights: torch.Tensor, a: float):
    s = torch.tanh(weights) ** 2
    return torch.sum((a - s) * s)


def get_loss(y_hat, y_true, criterion: nn.Module, a: float, b: float, ternary: bool, parameters: list[Parameter]):
    if not ternary:
        return criterion(y_hat, y_true)
    else:
        regularization_term = sum([R(param, a) for param in parameters])
        c = criterion(y_hat, y_true)
        return c + b * regularization_term


def train(train_loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: Optimizer, ternary: bool, a: float, b: float, device: str):
    '''
    Function for the training step of the training loop
    '''
    model.train()
    running_loss = 0
    
    for X, y in train_loader:
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
    
        # Forward pass
        y_hat, probs = model(X)
        loss = get_loss(y_hat, y, criterion, a, b, ternary, model.parameters())
        
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

@torch.no_grad()
def validate(valid_loader: DataLoader, model: nn.Module, criterion: nn.Module, device: str):
    '''
    Function for the validation step of the training loop
    '''
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    return model, epoch_loss


def training_loop(model: nn.Module, criterion: nn.Module, optimizer: Optimizer, train_loader: DataLoader, \
    valid_loader: DataLoader, epochs: int, device: str, ternary: bool, a: float=None, b: float=None, tracker:Tracker=None):
    '''
    Function defining the entire training loop
    '''

    tracker.loop_init(model)
    # Train model
    for epoch in range(epochs):
        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, ternary, a, b, device)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

        tracker.track_loss(train_loss, valid_loss, train_acc, valid_acc)
    
    train_losses, valid_losses = tracker.summarise()
    return model, optimizer, (train_losses, valid_losses)


def run(conf: Configuration, args: Namespace, tracker: Tracker):
    # check device
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'

    # Implementing LeNet-5
    torch.manual_seed(conf.seed)
    
    model = get_model(10, conf.ternary, conf.a, conf.b)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)

    train_loader = dataloading.get_mnist_dataloader(train=True, samples=conf.samples, shuffle=True, batch_size=conf.batch_size)
    valid_loader = dataloading.get_mnist_dataloader(train=False, shuffle=True, batch_size=conf.batch_size)

    model, optimizer, errs = training_loop(model, criterion, optimizer, train_loader, valid_loader, args.epochs, device, conf.ternary, conf.a, conf.b, tracker)
    return errs


def get_arguments():
    parser = ArgumentParser(description='Training Procedure for a NN on MNIST')
    parser.add_argument('--config', type=str, default='config_2')
    parser.add_argument('--save_path', type=str, required=False, default='run2')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=0)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    return args


def run_grid(grid: Grid, args: Namespace):
    if args.save_path is not None:
        args.save_path = 'runs/' + args.save_path

    if os.path.exists(args.save_path):
        # delete all files within that directory
        for file in os.listdir(args.save_path):
            os.remove(os.path.join(args.save_path, file))

    tracker = Tracker()
    tracker.add_logger(Progress(log_every=1))
    if (args.save_path is not None):
        tracker.add_logger(Errors(path=args.save_path, log_every=1))
        tracker.add_logger(Checkpoints(path=args.save_path, clear_dir=True, log_every=args.save_every if args.save_every >= 1 else args.epochs))
    if (args.plot):
        tracker.add_logger(Plotter())

    for idx, conf in enumerate(grid):
        print(conf)
        train_err, _ = run(conf, args, tracker)
        assert(len(train_err) == args.epochs)


if __name__ == '__main__':
    conf_path = 'configs.json'
    args = get_arguments()
    grid = read_grid(conf_path, args.config)
    run_grid(grid, args)

