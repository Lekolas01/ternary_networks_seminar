import inspect
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

class Logger:
    """ Extracts and/or persists tracker information. """
    def __init__(self, path: str = None, log_every: int=1):
        """
        Parameters
        ----------
        path : str or Path, optional
            Path to where data will be logged.
        """
        self.path = None if path is None else Path(path).expanduser().resolve() 
        self.log_every = log_every

    def log(self, epoch, update, train_loss, valid_loss, train_acc, valid_acc):
        """
        Log the loss and other metrics of the current mini-batch.

        Parameters
        ----------
        epoch 
            Rank of the current epoch.
        update 
            Rank of the current update.
        loss 
            Loss value of the current batch.
        """
        pass

    def log_summary(self):
        """
        Log the summary of metrics on the current epoch.

        Parameters
        ----------
        epoch 
            Rank of the current epoch.
        update 
            Rank of the current update.
        avg_loss 
            Summary value of the current epoch.
        """
        pass


class Tracker:
    """ Tracks useful information on the current epoch. """
    def __init__(self, *loggers: Logger):
        """
        Parameters
        ----------
        logger0, logger1, ... loggerN : Logger
            One or more loggers for logging training information.
        """
        self.epoch = 0
        self.train_losses = []
        self.valid_losses = []
        self.loggers = list(loggers)

    def add_logger(self, logger: Logger):
        if (logger not in self.loggers):
            self.loggers.append(logger)


    def track_loss(self, train_loss, valid_loss, train_acc, valid_acc):
        self.epoch += 1
        for logger in self.loggers:
            if (self.epoch % logger.log_every == 0):
                logger.log(self.epoch, train_loss, valid_loss, train_acc, valid_acc)
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)


    def summarise(self):
        for logger in self.loggers:
            logger.log_summary()
        return self.train_losses, self.valid_losses


class Progress(Logger):
    " Log progress of epoch to stdout. "

    def __init__(self, **base_args):
        """
        Parameters
        ----------
        """
        super().__init__(**base_args)


    def log(self, epoch, train_loss, valid_loss, train_acc, valid_acc):
        if (self.path is None):
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                f'Epoch: {epoch}\t'
                f'Train loss: {train_loss:.4f}\t'
                f'Valid loss: {valid_loss:.4f}\t'
                f'Train accuracy: {100 * train_acc:.2f}\t'
                f'Valid accuracy: {100 * valid_acc:.2f}')
        else:
            print("NotImplemented")
        
    def log_summary(self):
        pass


def get_methods(obj):
    return [m for m in dir(obj) if callable(getattr(obj, m)) and not m.startswith('_')]

def get_parameters(fn: callable):
    return inspect.getfullargspec(fn)


class Plotter(Logger):
    "Dynamically updates a loss plot after every epoch."
    def __init__(self, **base_args):
        super().__init__(**base_args)
        self.train_losses = []
        self.valid_losses = []
        self.epochs = []
    
    def log(self, epoch, train_loss, valid_loss, train_acc, valid_acc):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        if (len(self.epochs) == 1): 
            return
        
        if (len(self.epochs) == 2):
            plt.ion()
            plt.style.use('seaborn')
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set(title='Loss over Epochs', xlabel='Epoch', ylabel='Loss')
            self.train_line, = self.ax.plot(self.epochs, self.train_losses, label='Training Loss')
            self.valid_line, = self.ax.plot(self.epochs, self.valid_losses, label='Validation Loss')
            self.ax.legend()

        self.train_line.set_data(self.epochs, self.train_losses)
        self.valid_line.set_data(self.epochs, self.valid_losses)
        self.ax.set_xlim(min(self.epochs), max(self.epochs))
        self.ax.set_ylim(min(self.train_losses), max(self.train_losses))
        self.ax.set_xticks(self.epochs)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def log_summary(self):
        # don't close plot after last epoch
        plt.show(block=True)


class Checkpoints(Logger):
    DEFAULT_NAME = "config{epoch:03d}"
    EXT = ".pth"
    
    def __init__(self, network: nn.Module, **base_args):
        super().__init__(**base_args)
        self.network = network

        if self.path.is_dir() or not self.path.suffix:
            # assume path is directory
            self.path = self.path / Checkpoints.DEFAULT_NAME
        # assure correct extension
        self.path = self.path.with_suffix(Checkpoints.EXT)
        # create directory if necessary
        self.path.parent.mkdir(exist_ok=True, parents=True)


    def log(self, epoch, train_loss, valid_loss, train_acc, valid_acc):
        try:
            save_path = str(self.path).format(epoch=epoch)
            torch.save(self.network, save_path)
        except Exception as inst:
            print(f"Could not save model to {save_path}: {inst}")

    
    def log_summary(self):
        pass