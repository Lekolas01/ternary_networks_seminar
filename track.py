import inspect
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

class Logger:
    """ Extracts and/or persists tracker information. """
    def __init__(self, path: str = None):
        """
        Parameters
        ----------
        path : str or Path, optional
            Path to where data will be logged.
        """
        self.path = None if path is None else Path(path).expanduser().resolve() 

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
        self.losses = []
        self.loggers = list(loggers)

    def add_logger(self, logger: Logger):
        if (logger not in self.loggers):
            self.loggers.append(logger)


    def track_loss(self, train_loss, valid_loss, train_acc, valid_acc):
        self.epoch += 1
        for logger in self.loggers:
            logger.log(self.epoch, train_loss, valid_loss, train_acc, valid_acc)
        self.losses.append(train_loss)


    def summarise(self):
        res = sum(self.losses) / max(len(self.losses), 1)
        self.losses.clear()

        for logger in self.loggers:
            logger.log_summary()

        self.epoch += 1
        return res


class Progress(Logger):
    " Log progress of epoch to stdout. "

    def __init__(self, path: str=None):
        """
        Parameters
        ----------
        """
        super().__init__(path)


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
    def __init__(self, path: str=None):
        super().__init__(path)
        self.train_losses = []
    
    def log(self, epoch, train_loss, valid_loss, train_acc, valid_acc):
        self.train_losses.append(train_loss)
        if (len(self.train_losses) == 1): 
            return
        
        if (len(self.train_losses) == 2):
            plt.ion()
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.update({'xlabel': 'Epoch', 'ylabel':'Train Loss'})
            self.line, = self.ax.plot(self.train_losses)

        x = range(len(self.train_losses))
        self.line.set_data(x, self.train_losses)
        self.ax.set_xlim(0, len(self.train_losses) - 1)
        self.ax.set_ylim(min(self.train_losses), max(self.train_losses))
        self.ax.set_xticks(x)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def log_summary(self):
        # don't close plot after last epoch
        plt.show(block=True)


class Checkpoints(Logger):
    DEFAULT_NAME = "config{epoch:03d}"
    EXT = ".pth"
    
    def __init__(self, network: nn.Module, path: str = None):
        super().__init__(path)
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