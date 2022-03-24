from datetime import datetime
import matplotlib.pyplot as plt


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
        assert isinstance(self.log_every, int) and self.log_every >= 1

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
