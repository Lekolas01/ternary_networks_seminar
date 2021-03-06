import argparse
import os

import torch
import torch.nn as nn

import config
from track import *
from models.lenet5 import LeNet5, TernaryLeNet5
import utils

# ## Helper Functions

def get_accuracy(model, data_loader, device):
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


def R(weights: torch.Tensor, a):
    s = torch.tanh(weights) ** 2
    return torch.sum((a - s) * s)


def get_loss(y_hat, y_true, criterion, a, b,  ternary, parameters):
    if not ternary:
        return criterion(y_hat, y_true)
    else:
        regularization_term = sum([R(param, a) for param in parameters])
        c = criterion(y_hat, y_true)
        return c + b * regularization_term


def train(train_loader, model, criterion, optimizer, device, ternary, a, b):
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
        logits, probs = model(X)
        loss = get_loss(logits, y, criterion, a, b, ternary, model.parameters())
        
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

@torch.no_grad()
def validate(valid_loader, model, criterion, device):
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


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, ternary, a=None, b=None, tracker:Tracker=None):
    '''
    Function defining the entire training loop
    '''

    tracker.track_init()
    # Train model
    for epoch in range(epochs):
        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device, ternary, a, b)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)

        tracker.track_loss(train_loss, valid_loss, train_acc, valid_acc)
    
    train_losses, valid_losses = tracker.summarise()
    return model, optimizer, (train_losses, valid_losses)


def run(conf):
    # check device
    device = 'cuda' if not conf.no_cuda and torch.cuda.is_available() else 'cpu'

    train_loader = utils.get_mnist_dataloader(train=True, samples=conf.samples, shuffle=True, batch_size=conf.batch_size)
    valid_loader = utils.get_mnist_dataloader(train=False, shuffle=True, batch_size=conf.batch_size)

    # Implementing LeNet-5
    torch.manual_seed(conf.seed)
    
    model = LeNet5(10) if not conf.ternary else TernaryLeNet5(10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    progress = Progress(model, conf.save_path + "/logs" if conf.save_path else None)
    tracker = Tracker(progress)
    if (hasattr(conf, 'plot') and conf.plot):
        tracker.add_logger(Plotter())
    if (hasattr(conf, 'save_path') and conf.save_path is not None):
        tracker.add_logger(Checkpoints(model=model, path=conf.save_path, log_every=conf.save_every if hasattr(conf, 'save_every') else 1))

    model, optimizer, errs = training_loop(model, criterion, optimizer, train_loader, valid_loader, conf.epochs, device, conf.ternary, conf.a, conf.b, tracker=tracker)
    return errs


def get_configuration(config_path, prop: str, consider_cmd_args=True):
    conf = getattr(config.read_config(config_path, yaml=False), prop)
    if (consider_cmd_args):
        parser = argparse.ArgumentParser(description='Training Procedure for LeNet on MNIST')
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--no_cuda', action='store_true', required=False)
        parser.add_argument('--samples', type=int, default=conf.samples)
        parser.add_argument('--epochs', type=int, default=conf.epochs)
        parser.add_argument('--batch_size', type=int, default=conf.batch_size)
        parser.add_argument('--lr', type=float, default=conf.lr)

        parser.add_argument('--save_path', required=False, type=str)
        parser.add_argument('--save_every', required=False, type=int, default=1)
        parser.add_argument('--plot', required=False, action='store_true')
        parser.add_argument('--plot_every', required=False, type=int, default=1)

        parser.add_argument('--ternary', action='store_true', default = conf.ternary)
        parser.add_argument('--a', type=float, default = conf.a)
        parser.add_argument('--b', type=float, default = conf.b)
        args = parser.parse_args()
        
        for arg in (arg for arg in dir(args) if not arg.startswith('_')):
            if (hasattr(conf, arg)):
                conf.overwrite(arg, getattr(args, arg))
            else:
                conf.__setitem__(arg, getattr(args, arg))
    return conf


def clear_directory(dir_path):
    if not os.path.isdir(dir_path): return
    for file in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, file))


def grid_search(conf_path, prop):
    grid = config.read_grid(conf_path, prop)
    for idx, conf in enumerate(grid):
        conf.save_path = 'runs/' + prop + '/' + str(idx)
        print(conf)
        # clear directory
        clear_directory(conf.save_path)
        train_err, _ = run(conf)
        assert(len(train_err) == conf.epochs)


def single_run(conf_path, prop):
    conf = get_configuration(conf_path, prop)

    print("Executing run with the following configuration:")
    print("\t       Variable |\tValue")
    print("\t--------------------------")
    for arg in conf:
        print(f"\t{arg:>15} | {str(getattr(conf, arg)):>11}")

    #train model
    train_err, valid_err = run(conf)
    assert(len(train_err) == conf.epochs)


if __name__ == '__main__':
    #single_run('configs.json', 'single_run')
    grid_search('configs.json', 'grid_search')
 
