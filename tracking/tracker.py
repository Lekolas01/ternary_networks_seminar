import numpy as np
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from models.ternary import TernaryModule
import utils
from config import Configuration
from .loggers import Logger


class Tracker:
    COMPLEXITY_LIMIT = 200

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
        valid_acc = train_acc
        #valid_acc = utils.get_accuracy(self.model, self.valid_loader, self.device)

        # mean distance from full-precision and sparsity
        weights = np.tanh(utils.get_all_weights(self.model).detach().cpu().numpy())
        distance, sparsity = utils.distance_from_int_precision(weights)

        # train- and test accuracies after quantization
        if isinstance(self.model, TernaryModule):
            quantized_model = self.model.quantized(simplify=False).to(self.device)
            compl = quantized_model.complexity()
            simple_model = self.model.quantized(simplify=True).to(self.device)
            simple_compl = simple_model.complexity()
            q_train_acc = utils.get_accuracy(quantized_model, self.train_loader, self.device).item()
            q_valid_acc = q_train_acc
            #q_valid_acc = utils.get_accuracy(quantized_model, self.valid_loader, self.device).item()
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
        return simple_compl >= self.COMPLEXITY_LIMIT


    def summarise(self):
        for logger in self.loggers:
            logger.log_summary()
        return self.train_losses, self.valid_losses
