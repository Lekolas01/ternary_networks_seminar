from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch
import torch.nn as nn

from config import Grid, read_grid
from tracking import Tracker, Progress, Results, Plotter
from models.factory import ModelFactory
from dataloading import DataloaderFactory
from train_model import training_loop


def run(seed: int, data: str, ternary: bool, batch_size: int, lr: float, schedule_lr: bool, epochs: int, a: float=None, b: float=None, tracker=Tracker(), **kwargs) -> list[float]:
    torch.manual_seed(seed)

    # check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ModelFactory(data, ternary, a, b).to(device)

    train_loader = DataloaderFactory(ds=data, train=True, shuffle=True, batch_size=batch_size)
    valid_loader = DataloaderFactory(ds=data, train=False, shuffle=True, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss if data == 'mnist' else nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if (schedule_lr):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[100, 160], gamma=0.1)

    return training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, tracker, scheduler, **kwargs)


def get_arguments() -> Namespace:
    parser = ArgumentParser(description='Generic Training Procedure for a NN with parameters specified in a configuration.')
    parser.add_argument('config', nargs=1, help='The name of the training configuration.')
    parser.add_argument('-s', '--save_path', type=str, required=False, help='If specified, will save the trained model and training information to this path.')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='For how many epochs you want to train.')
    parser.add_argument('--save_every', type=int, default=1, help='After how many epochs you want to create checkpoints. Only meaningful when saving the model.')
    parser.add_argument('-p', '--plot', action='store_true', help='If true, will plot the current loss at each iteration.')
    args = parser.parse_args()
    return args


def run_grid(grid: Grid, args: Namespace, tracker: Tracker):
    for idx, conf in enumerate(grid):
        print(idx, ':', conf)
        train_err, _ = run(
            seed=conf.seed, data=conf.data, ternary=conf.ternary,
            batch_size=conf.batch_size, lr=conf.lr, schedule_lr=conf.schedule_lr,
            epochs=args.epochs, a=conf.a, b=conf.b, tracker=tracker
        )
        assert(len(train_err) == args.epochs)


if __name__ == '__main__':
    conf_path = 'configs.json'
    args = get_arguments()
    grid = read_grid(conf_path, args.config[0])

    tracker = Tracker()
    tracker.add_logger(Progress(log_every=1))
    if (args.save_path is not None):
        save_path = Path('runs') / args.save_path
        tracker.add_logger(Results(path=save_path, model_every=args.epochs))
    if (args.plot):
        tracker.add_logger(Plotter())

    run_grid(grid, args, tracker)
