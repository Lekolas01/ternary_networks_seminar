from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import utils
from torch.utils.data.dataloader import DataLoader
from models.ternary import TernaryModule
from statistics import mean


class Logger:
    """Extracts and/or persists tracker information."""

    def __init__(self, log_every: int = 1):
        """
        Parameters
        ----------
        log_every: int, optional
            After how many epochs you want to log.
        """
        self.log_every = log_every
        assert isinstance(self.log_every, int) and self.log_every >= 1

    def _register_tracker(self, tracker):
        self.t = tracker

    def training_start(self):
        pass

    def epoch_start(self, **kwargs):
        pass

    def batch_start(self):
        pass

    def batch_end(self):
        pass

    def epoch_end(self, **kwargs):
        """
        Enables the Logger to add a hook at the end of every epoch during training.

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

    def training_end(self):
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


class LogMetrics(Logger):
    "Log Losses of epoch to stdout."

    def __init__(self, metrics: list[str], **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics

    def epoch_end(self):
        self.m_format = {
            "timestamp": f"{datetime.now().time().replace(microsecond=0)} ----",
            "epoch": f"Epoch: {self.t.epoch}",
            "train_loss": f"Loss: {self.t.mean_train_losses[-1]:.4f}",
            "train_acc": f"Train: {(self.t.train_acc * 100):.2f}%",
            "valid_acc": f"Valid: {(self.t.valid_acc * 100):.2f}%",
        }

        for metric in self.metrics:
            print(f"{self.m_format[metric]}", end="\t")
        print()


class Plotter(Logger):
    "Dynamically updates a loss plot after every epoch."

    def __init__(self, **base_args):
        super().__init__(**base_args)
        self.train_losses = []
        self.valid_losses = []
        self.epochs = []

    def epoch_end(self, epoch, train_loss, valid_loss, **kwargs):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        if len(self.epochs) == 1:
            return

        if len(self.epochs) == 2:
            plt.ion()
            plt.style.use("seaborn")
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            self.ax.set(title="Loss over Epochs", xlabel="Epoch", ylabel="Loss")
            (self.train_line,) = self.ax.plot(
                self.epochs, self.train_losses, label="Training Loss"
            )
            (self.valid_line,) = self.ax.plot(
                self.epochs, self.valid_losses, label="Validation Loss"
            )
            self.ax.legend()

        self.train_line.set_data(self.epochs, self.train_losses)
        self.valid_line.set_data(self.epochs, self.valid_losses)
        self.ax.set_xlim(min(self.epochs), max(self.epochs))
        self.ax.set_ylim(min(self.train_losses), max(self.train_losses))
        self.ax.set_xticks(self.epochs)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def training_end(self):
        # block=True so as to not close plot after the last epoch
        plt.show(block=True)


class LogModel(Logger):
    "Log current model."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def training_start(self, **kwargs):
        # print model at the start of training
        # from train_model import validate
        # validate(self.t.valid_loader, self.t.model, self.t.criterion, self.t.device)
        # print(f"Model :\t{self.t.model}")
        pass

    def epoch_start(self, **kwargs):
        pass

    def epoch_end(self, **kwargs):
        print(self.t.model)

    def batch_start(self):
        pass
        # print(f"Model :\t{self.t.model}")


class Tracker:
    """Tracks useful information on the current epoch."""

    def __init__(self, *loggers: Logger):
        """
        Parameters
        ----------
        logger0, logger1, ... loggerN : Logger
            List of loggers used for logging training information.
        """
        self.loggers = []
        for logger in loggers:
            self.add_logger(logger)

    def add_logger(self, logger: Logger):
        if logger not in self.loggers:
            self.loggers.append(logger)
            logger._register_tracker(self)

    def training_start(
        self,
        model: nn.Module,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        loss_fn: nn.Module,
    ):
        self.model = model
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.loss_fn = loss_fn

        self.device = next(self.model.parameters()).device
        self.epoch = 0
        self.mean_train_losses = []  # the mean of every epochs training loss
        self.mean_valid_losses = []  # the mean of every epochs validation loss

        for logger in self.loggers:
            logger.training_start()

    def epoch_start(self):
        for logger in self.loggers:
            logger.epoch_start()

    def batch_start(self):
        for logger in self.loggers:
            logger.batch_start()

    def batch_end(self):
        for logger in self.loggers:
            logger.batch_end()

    def epoch_end(self, train_losses, valid_losses):
        self.epoch += 1

        # calculate important metrics:
        # train- and test accuracy
        self.mean_train_losses.append(mean(train_losses))
        self.mean_valid_losses.append(mean(valid_losses))
        self.train_acc = utils.accuracy(self.model, self.train_dl, self.device)
        self.valid_acc = self.train_acc
        # valid_acc = utils.accuracy(self.model, self.valid_loader, self.device)

        # mean distance from full-precision and sparsity
        weights = np.tanh(utils.get_all_weights(self.model).detach().cpu().numpy())
        distance, sparsity = utils.distance_from_int_precision(weights)

        # train- and test accuracies after quantization
        if isinstance(self.model, TernaryModule):
            quantized_model = self.model.quantized(prune=False).to(self.device)
            self.compl = quantized_model.complexity()
            simple_model = self.model.quantized(prune=True).to(self.device)
            simple_compl = simple_model.complexity()
            self.q_train_acc = utils.accuracy(
                quantized_model, self.train_dl, self.device
            )
            self.q_valid_acc = self.q_train_acc
            # q_valid_acc = utils.accuracy(quantized_model, self.valid_loader, self.device).item()
        else:
            self.q_train_acc = self.q_valid_acc = self.compl = self.simple_compl = 0.0

        for logger in self.loggers:
            if self.epoch % logger.log_every == 0:
                # pass on the calculated metrics to all subclasses, so they can print it or whatever they want to do with it.
                logger.epoch_end()

        return self.simple_compl

    def training_end(self) -> tuple[list[float], list[float]]:
        for logger in self.loggers:
            logger.training_end()
        return self.mean_train_losses, self.mean_valid_losses