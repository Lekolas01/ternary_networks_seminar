from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import utils
import numpy as np
import os

import torch
import torch.nn as nn


class Logger:
    """ Extracts and/or persists tracker information. """
    def __init__(self, log_every: int=1):
        """
        Parameters
        ----------
        log_every: int, optional
            After how many epochs you want to log
        """
        self.log_every = log_every

    def log_init(self):
        pass

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
        self.loggers = list(loggers)
        self.idx = 0

    def add_logger(self, logger: Logger):
        if (logger not in self.loggers):
            self.loggers.append(logger)


    def loop_init(self, model):
        self.model = model
        self.idx += 1
        self.epoch = 0
        self.train_losses = []
        self.valid_losses = []

        for logger in self.loggers:
            logger.log_init()


    def track_loss(self, train_loss, valid_loss, train_acc, valid_acc):
        assert(isinstance(self.model, nn.Module))
        self.epoch += 1
        for logger in self.loggers:
            if (self.epoch % logger.log_every == 0):
                logger.log(self.idx, self.model, self.epoch, train_loss, valid_loss, train_acc, valid_acc)
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


    def log_init(self):
        pass


    def log(self, idx, model, epoch, train_loss, valid_loss, train_acc, valid_acc):
        weights = utils.get_all_weights(model).detach().cpu().numpy()
        distance = np.mean(utils.distance_from_int_precision(weights))
        print(f'{datetime.now().time().replace(microsecond=0)} --- '
            f'Epoch: {epoch}\t'
            f'Train loss: {train_loss:.4f}\t'
            f'Valid loss: {valid_loss:.4f}\t'
            f'Train accuracy: {100 * train_acc:.2f}\t'
            f'Valid accuracy: {100 * valid_acc:.2f}\t'
            f'Distance: {distance:.4f}')
            
        
    def log_summary(self):
        pass


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


class FileLogger(Logger):
    def __init__(self, path: str, clear_dir: bool=False, **base_args):
        super().__init__(**base_args)
        self.path = None if path is None else Path(path).expanduser().resolve()
        # create directory if necessary
        self.path.mkdir(exist_ok=True, parents=True)
        assert(os.path.isdir(str(self.path)))


class Errors(FileLogger):
    LINE_FORMAT = "{idx},{epoch},{tl:.4f},{vl:.4f},{ta:.4f},{va:.4f},{d:.4f}\n"
    FILE_NAME = "errors.csv"

    def __init__(self, **base_args):
        super().__init__(**base_args)
        self.errors_path = self.path / self.FILE_NAME

        # add header to errors file
        head = ['idx', 'epoch', 'train_loss', 'valid_loss', 'train_acc', 'valid_acc', 'distance']
        with open(self.errors_path, 'w') as f:
            f.write(','.join(head) + '\n')


    def log(self, idx, model, epoch, train_loss, valid_loss, train_acc, valid_acc):
        weights = utils.get_all_weights(model).detach().cpu().numpy()
        distance = np.mean(utils.distance_from_int_precision(weights))

        log_line = self.LINE_FORMAT \
            .format(idx=idx, epoch=epoch, tl=train_loss, vl=valid_loss, ta=train_acc, va=valid_acc, d=distance)
        with open(self.errors_path, 'a') as f:
            f.write(log_line)


class Checkpoints(FileLogger):
    DEFAULT_NAME = "config{idx:02d}_epoch{epoch:03d}"
    EXT = ".pth"
    
    def __init__(self, **base_args):
        super().__init__(**base_args)
        self.model_path = (self.path / Checkpoints.DEFAULT_NAME).with_suffix(Checkpoints.EXT)


    def log(self, idx, model, epoch, train_loss, valid_loss, train_acc, valid_acc):
        try:
            save_path = str(self.model_path).format(idx=idx, epoch=epoch)
            torch.save(model, save_path)
        except Exception as inst:
            print(f"Could not save model to {save_path}: {inst}")
