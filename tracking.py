from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import utils
import numpy as np

import torch
import torch.nn as nn
from config import Configuration


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

    def loop_init(self, idx: int, conf: Configuration):
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


    def loop_init(self, model: nn.Module, **kwargs):
        self.model = model
        self.idx += 1
        self.epoch = 0
        self.train_losses = []
        self.valid_losses = []

        for logger in self.loggers:
            logger.loop_init(self.epoch, **kwargs)


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

    def loop_init(self, epoch: int, **kwargs):
        pass

    def log(self, idx, model, epoch, train_loss, valid_loss, train_acc, valid_acc):
        weights = np.tanh(utils.get_all_weights(model).detach().cpu().numpy())
        d, (minus_ones, zeros, plus_ones) = utils.distance_from_int_precision(weights)
        distance = np.mean(d)
        print(f'{datetime.now().time().replace(microsecond=0)} --- '
            f'Epoch: {epoch}\t'
            f'Train loss: {train_loss:.4f}\t'
            f'Valid loss: {valid_loss:.4f}\t'
            f'Train accuracy: {100 * train_acc:.2f}\t'
            f'Valid accuracy: {100 * valid_acc:.2f}\t'
            f'Distance: {distance:.4f}\t'
            f'Spread: {minus_ones:.2f} {zeros:.2f} {plus_ones:.2f}\t')
        
    def log_summary(self):
        pass


class Plotter(Logger):
    "Dynamically updates a loss plot after every epoch."
    def __init__(self, **base_args):
        super().__init__(**base_args)
        self.train_losses = []
        self.valid_losses = []
        self.epochs = []
    
    def log(self, idx, model, epoch, train_loss, valid_loss, train_acc, valid_acc):
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


class ResultsLogger(Logger):
    ERROR_FILE = "errors.csv"
    ERRORS_FORMAT = "{idx},{epoch},{tl:.4f},{vl:.4f},{ta:.4f},{va:.4f},{d:.4f},{m:.4f},{z:.4f},{p:.4f}\n"

    CONFIGS_FILE = "configs.csv"
    CONFIGS_FORMAT = "{seed},{lr},{batch_size},{samples},{ternary},{a},{b}\n"

    MODEL_FILE = "config{idx:02d}_epoch{epoch:03d}.pth"

    def __init__(self, path: str, model_every=1):
        super().__init__(log_every = 1)
        assert(path is not None)
        self.path = Path(path)
        self.model_every = model_every

        self.errors_path = self.path / self.ERROR_FILE
        self.configs_path = self.path / self.CONFIGS_FILE
        self.model_path = self.path / self.MODEL_FILE

        self.path.mkdir(exist_ok=False, parents=True)

        # prepare configs file
        with open(self.configs_path, 'w') as f:
            f.write(self.CONFIGS_FORMAT.format(s='seed', lr='lr', bs='batch_size', sa='samples', t='ternary', a='a', b='b'))
        
        # prepare errors file
        with open(self.errors_path, 'w') as f:
            f.write('idx,epoch,train_loss,valid_loss,train_acc,valid_acc,distance,minus_ones,zeros,plus_ones\n')


    def loop_init(self, idx: int, seed, lr, batch_size, samples, ternary, a, b, **kwargs):
        # log the configuration
        with open(self.configs_path, 'a') as f:
            f.write(self.CONFIGS_FORMAT.format(seed, lr, batch_size, samples, ternary, a, b))


    def log(self, idx, model, epoch, train_loss, valid_loss, train_acc, valid_acc):
        weights = np.tanh(utils.get_all_weights(model).detach().cpu().numpy())
        d, (minus_ones, zeros, plus_ones) = utils.distance_from_int_precision(weights)
        distance = np.mean(d)

        # log errors
        log_line = self.ERRORS_FORMAT \
            .format(idx=idx, epoch=epoch, tl=train_loss, vl=valid_loss, ta=train_acc, va=valid_acc, d=distance, m=minus_ones, z=zeros, p=plus_ones)
        with open(self.errors_path, 'a') as f:
            f.write(log_line)

        if epoch % self.model_every == 0:
            print(self.model_every, idx)
            # log model
            try:
                curr_model_path = str(self.model_path).format(idx=idx, epoch=epoch)
                torch.save(model, curr_model_path)
            except Exception as inst:
                print(f"Could not save model to {curr_model_path}: {inst}")

