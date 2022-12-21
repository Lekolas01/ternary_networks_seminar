import numpy as np
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from models.ternary import TernaryModule
import utils
from config import Configuration
from .loggers import Logger


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


    def distance_from_int_precision(x: np.ndarray):
        """
        Calculates the avg. distance of x w.r.t. the nearest integer. 
        x must be in range [-1, 1]
        When doing many update steps with WDR Regularizer, this distance should decrease.
        """
        assert(np.allclose(x, np.zeros_like(x), atol = 1, rtol=0))
        ans = np.full_like(x, 2)
        bases = [-1, 0, 1]
        for base in bases:
            d = np.abs(x - base)
            ans = np.where(d < ans, d, ans)
        minus_ones = np.count_nonzero(x <= -0.5) / x.shape[0]
        plus_ones = np.count_nonzero(x >= 0.5) / x.shape[0]
        zeros = 1 - minus_ones - plus_ones
        return np.mean(ans), zeros


    def track_loss(self, train_loss, valid_loss):
        self.epoch += 1

        # calculate important metrics:
        # train- and test accuracy
        train_acc = utils.accuracy(self.model, self.train_loader, self.device)
        valid_acc = train_acc
        #valid_acc = utils.accuracy(self.model, self.valid_loader, self.device)

        # mean distance from full-precision and sparsity
        weights = np.tanh(utils.get_all_weights(self.model).detach().cpu().numpy())
        distance, sparsity = utils.distance_from_int_precision(weights)

        # train- and test accuracies after quantization
        if isinstance(self.model, TernaryModule):
            quantized_model = self.model.quantized(prune=False).to(self.device)
            compl = quantized_model.complexity()
            simple_model = self.model.quantized(prune=True).to(self.device)
            simple_compl = simple_model.complexity()
            q_train_acc = utils.accuracy(quantized_model, self.train_loader, self.device).item()
            q_valid_acc = q_train_acc
            #q_valid_acc = utils.accuracy(quantized_model, self.valid_loader, self.device).item()
        else:
            q_train_acc, q_valid_acc, compl, simple_compl = 0.0, 0.0, 0.0, 0.0

        for logger in self.loggers:
            if (self.epoch % logger.log_every == 0):
                # pass on the calculated metrics to all subclasses, so they can print it or whatever they want to do with it.
                logger.log(
                    train_loss=train_loss, valid_loss=valid_loss, 
                    train_acc=train_acc, valid_acc=valid_acc, 
                    distance=distance, sparsity=sparsity, 
                    q_train_acc=q_train_acc, q_valid_acc=q_valid_acc, 
                    compl=compl, simple_compl=simple_compl)

        self.train_losses.append(train_loss)
        self.valid_losses.append(valid_loss)
        return simple_compl


    def summarise(self):
        for logger in self.loggers:
            logger.log_summary()
        return self.train_losses, self.valid_losses
