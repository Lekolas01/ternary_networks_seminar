import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import config
from models.lenet5 import LeNet5
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


def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    plt.show(block=False)
    
    # change the plot style to default
    plt.style.use('default')


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        X = X.to(device)
        y_true = y_true.to(device)
    
        # Forward pass
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss


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


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    train_losses = []
    valid_losses = []
 
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)
    
    return model, optimizer, (train_losses, valid_losses)


def run(conf):
    # check device
    device = 'cuda' if not conf.no_cuda and torch.cuda.is_available() else 'cpu'

    train_loader = utils.get_mnist_dataloader(True, shuffle=True, batch_size=conf.batch_size)
    valid_loader = utils.get_mnist_dataloader(False, shuffle=True, batch_size=conf.batch_size)

    # Implementing LeNet-5
    torch.manual_seed(conf.seed)

    model = LeNet5(conf.n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"start training loop (device = {device})...")
    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, conf.n_epochs, device)
    if (conf.save_path is not None):
        try:
            torch.save(model, conf.save_path)
            print(f"Saved the trained model to {conf.save_path}.")
        except:
            print("Could not save model to {conf.save_path}.")
    print("run done.")


def get_configuration(config_path, consider_cmd_args=True):
    conf = config.read_config(config_path, yaml=False).lenet5
    if (consider_cmd_args):
        parser = argparse.ArgumentParser(description='Training Procedure for LeNet on MNIST')
        parser.add_argument('--n_epochs', type=int, default=conf.n_epochs)
        parser.add_argument('--n_train_samples', type=int, default=conf.n_train_samples)
        parser.add_argument('--lr', type=float, default=conf.lr)
        parser.add_argument('--batch_size', type=int, default=conf.batch_size)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--no_cuda', action='store_true', default=False)
        parser.add_argument('--save_path', required=False, type=str)
        
        args = parser.parse_args()
        
        for arg in (arg for arg in dir(args) if not arg.startswith('_')):
            if (hasattr(conf, arg)):
                conf.overwrite(arg, getattr(args, arg))
            else:
                conf.__setitem__(arg, getattr(args, arg))


    print("Executing run with the following configuration:")
    print("\t\tName\t|\tValue")
    print("\t--------------------------")
    for arg in conf:
        print(f"\t{arg:>15} | {str(getattr(conf, arg)):>11}")

if __name__ == '__main__':
    conf = get_configuration('configs.json')
    run(conf)
