import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config
from lenet5_model import LeNet5


# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 64
N_TRAIN_SAMPLES = 10000
N_EPOCHS = 2

N_CLASSES = 10

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define transforms
    # transforms.ToTensor() automatically scales the images to [0,1] range
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor()])

    # download and create datasets
    train_dataset = datasets.MNIST(root='data/', 
                                train=True, 
                                transform=transform,
                                download=True)
    if (1 <= conf.n_train_samples < len(train_dataset)):
        train_dataset = data_utils.Subset(train_dataset, torch.arange(conf.n_train_samples))

    valid_dataset = datasets.MNIST(root='data/', 
                                train=False, 
                                transform=transform)

    # define the data loaders
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=conf.batch_size, 
                            shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset, 
                            batch_size=conf.batch_size, 
                            shuffle=False)

    # Implementing LeNet-5
    torch.manual_seed(conf.seed)

    model = LeNet5(conf.n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    criterion = nn.CrossEntropyLoss()

    print("start training loop...")
    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, conf.n_epochs, device)
    print("training loop done.")


if __name__ == '__main__':
    conf = config.read_config('configs.json', yaml=False).lenet5
    parser = argparse.ArgumentParser(description='Training Procedure for LeNet on MNIST')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False)
    args = parser.parse_args()
    
    for arg in (arg for arg in dir(args) if not arg.startswith('_')):
        conf.__setitem__(arg, getattr(args, arg))


    print("Executing run with the following configuration:")
    print("\t\tName\t|\tValue")
    print("\t--------------------------")
    for arg in conf:
        print(f"\t{arg:>15} | {getattr(conf, arg):>11}")
    
    run(conf)