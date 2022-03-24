from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from models.ternary import TernaryModule
import utils
from config import Configuration


class Logger:
    """ Extracts and/or persists tracker information. """
    def __init__(self, log_every: int=1):
        """
        Parameters
        ----------
        log_every: int, optional
            After how many epochs you want to log.
        """
        self.log_every = log_every

    def loop_init(self):
        pass

    def log(self, **kwargs):
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


class Progress(Logger):
    " Log progress of epoch to stdout. "

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        """
        super().__init__(**kwargs)

    def loop_init(self, **kwargs):
        super().loop_init(**kwargs)

    def log(self, train_loss, train_acc, valid_acc, q_train_acc, q_valid_acc, distance, sparsity, **kwargs):

        print(f'{datetime.now().time().replace(microsecond=0)} --- '
            f'Epoch: {self.t.epoch}\t'
            f'Loss: {train_loss:.4f}  \t'
            f'Train: {100 * train_acc:.4f}  \t'
            f'Valid: {100 * valid_acc:.4f}  \t'
            f'Q-Train: {100 * q_train_acc:.4f}  \t'
            f'Q-Valid: {100 * q_valid_acc:.4f}  \t'
            f'Distance: {distance:.4f}\t'
            f'Sparsity: {sparsity:.4f}\t')
        
    def log_summary(self):
        pass


class Plotter(Logger):
    "Dynamically updates a loss plot after every epoch."
    def __init__(self, **base_args):
        super().__init__(**base_args)
        self.train_losses = []
        self.valid_losses = []
        self.epochs = []
    
    def log(self, epoch, train_loss, valid_loss, **kwargs):
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

    class Errors(Logger):
        ERROR_FILE = "errors.csv"
        ERRORS_HEADER_FORMAT = "{idx},{epoch},{tl},{vl},{ta},{va},{qta},{qva},{dist},{sp}\n"
        ERRORS_FORMAT = "{idx},{epoch},{tl:.4f},{vl:.4f},{ta:.4f},{va:.4f},{qta:.4f},{qva:.4f},{dist:.4f},{sp:.4f}\n"

        def __init__(self, checkpoints, **kwargs):
            super().__init__(**kwargs)
            self.cp = checkpoints
            self.errors_path = self.cp.path / self.ERROR_FILE

            # prepare errors file
            with open(self.errors_path, 'w') as f:
                new_var = self.ERRORS_HEADER_FORMAT.format(
                    idx='idx', epoch='epoch',
                    tl='train_loss', vl='valid_loss',
                    ta='train_acc', va='valid_acc',
                    qta='quantized_train_acc', qva='quantized_valid_acc',
                    dist='distance', sp='sparsity')
                print(f)
                print(new_var)

                f.write(new_var)


        def log(self, train_loss, valid_loss, train_acc, valid_acc, q_train_acc, q_valid_acc, distance, sparsity, **kwargs):
            # log errors
            with open(self.errors_path, 'a') as f:
                f.write(self.ERRORS_FORMAT.format(
                    idx=self.cp.t.conf_idx, epoch=self.cp.t.epoch,
                    tl=train_loss, vl=valid_loss,
                    ta=train_acc, va=valid_acc,
                    qta=q_train_acc, qva=q_valid_acc,
                    dist=distance, sp=sparsity
                ))

        
    class Configs(Logger):
        CONFIGS_FILE = "configs.csv"
        CONFIGS_FORMAT = "{seed},{lr},{batch_size},{ternary},{a},{b}\n"

        def __init__(self, checkpoints, **kwargs):
            super().__init__(**kwargs)
            self.cp = checkpoints
            self.configs_path = self.cp.path / self.CONFIGS_FILE

            # prepare configs file
            with open(self.configs_path, 'w') as f:
                f.write(self.CONFIGS_FORMAT.format(
                    seed='seed', lr='learning_rate', 
                    batch_size='batch_size', ternary='ternary', 
                    a='a', b='b'))

        def loop_init(self, **kwargs):
            conf = self.cp.t.conf
            # log the configuration
            with open(self.configs_path, 'a') as f:
                f.write(self.CONFIGS_FORMAT.format(seed=conf.seed, lr=conf.lr, batch_size=conf.batch_size, ternary=conf.ternary, a=conf.a, b=conf.b))


    class Models(Logger):
        MODEL_FILE = "config{idx:02d}_epoch{epoch:03d}.pth"

        def __init__(self, checkpoints, **kwargs):
            super().__init__(**kwargs)
            self.cp = checkpoints
            self.model_path = self.cp.path / self.MODEL_FILE

        def log(self, **kwargs):
            try:
                curr_model_path = str(self.model_path).format(idx=self.t.idx, epoch=self.t.epoch)
                torch.save(self.t.model, curr_model_path)
            except Exception as inst:
                print(f"Could not save model to {curr_model_path}: {inst}")


    def __init__(self, path: str, **kwargs):
        super().__init__(log_every=1)
        self.path = Path(path)
        assert(path is not None)
        self.path.mkdir(exist_ok=True, parents=True)

        self.error_logger = self.Errors(self)
        self.config_logger = self.Configs(self)
        self.model_logger = self.Models(self)
        self.loggers = [self.error_logger, self.config_logger]


    def loop_init(self):
        super().loop_init()
        for logger in self.loggers:
                logger.loop_init()

    def log(self, **kwargs):
        for logger in self.loggers:
            logger.log(**kwargs)


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
        self.conf_idx = 0

    def add_logger(self, logger: Logger):
        if (logger not in self.loggers):
            self.loggers.append(logger)
            logger.t = self


    def loop_init(self, model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, conf: Configuration, **kwargs):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.conf = conf
        self.conf_idx += 1
        self.epoch = 0
        self.train_losses = []
        self.valid_losses = []

        for logger in self.loggers:
            logger.loop_init(**kwargs)


    def track_loss(self, train_loss, valid_loss):
        self.epoch += 1

        # calculate important metrics:
        # train- and test accuracy
        train_acc = utils.get_accuracy(self.model, self.train_loader, self.device)
        valid_acc = utils.get_accuracy(self.model, self.valid_loader, self.device)

        # mean distance from full-precision and sparsity
        weights = np.tanh(utils.get_all_weights(self.model).detach().cpu().numpy())
        distance, sparsity = utils.distance_from_int_precision(weights)

        # train- and test accuracies after quantization
        if isinstance(self.model, TernaryModule):
            quantized_model = self.model.quantized().to(self.device)
            q_train_acc = utils.get_accuracy(quantized_model, self.train_loader, self.device)
            q_valid_acc = utils.get_accuracy(quantized_model, self.valid_loader, self.device)
        else:
            q_train_acc, q_valid_acc = 0.0, 0.0

        for logger in self.loggers:
            if (self.epoch % logger.log_every == 0):
                # pass on the calculated metrics to all subclasses, so they can print it or whatever they want to do with it.
                logger.log(
                    train_loss=train_loss, valid_loss=valid_loss, 
                    train_acc=train_acc, valid_acc=valid_acc, 
                    distance=distance, sparsity=sparsity, 
                    q_train_acc=q_train_acc, q_valid_acc=q_valid_acc)

        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)


    def summarise(self):
        for logger in self.loggers:
            logger.log_summary()
        return self.train_losses, self.valid_losses
