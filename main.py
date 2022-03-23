from argparse import *
import torch
import torch.nn as nn

from config import Grid, Configuration, read_grid
from tracking import *
from models.factory import ModelFactory
from dataloading import DataloaderFactory
from train_model import *


def run(conf: Configuration, args: Namespace, tracker: Tracker) -> list[float]:
    torch.manual_seed(conf.seed)

    if not conf.ternary:
        conf.overwrite('a', 0.0)
        conf.overwrite('b', 0.0)

    # check device
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    
    model = ModelFactory(conf.data, conf.ternary, conf.a, conf.b).to(device)

    train_loader = DataloaderFactory(ds=conf.data, train=True, shuffle=True, batch_size=conf.batch_size)
    valid_loader = DataloaderFactory(ds=conf.data, train=False, shuffle=True, batch_size=conf.batch_size)

    criterion = nn.CrossEntropyLoss if conf.data == 'mnist' else nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    scheduler = None
    if (hasattr(conf, "schedule_lr") and conf.schedule_lr == True):
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[100, 160], gamma=0.1)

    return training_loop(model, criterion, optimizer, train_loader, valid_loader, args.epochs, device, tracker, scheduler, seed=conf.seed)


def get_arguments() -> Namespace:
    parser = ArgumentParser(description='Training Procedure for a NN on MNIST')
    parser.add_argument('config', nargs=1)
    parser.add_argument('-s', '--save_path', type=str, required=False)
    parser.add_argument('-e', '--epochs', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')
    args = parser.parse_args()
    return args


def run_grid(grid: Grid, args: Namespace):
    tracker = Tracker()
    tracker.add_logger(Progress(log_every=1))
    if (args.save_path is not None):
        save_path = Path('runs') / args.save_path
        tracker.add_logger(ResultsLogger(path=save_path, model_every=args.epochs))
    if (args.plot):
        tracker.add_logger(Plotter())

    for idx, conf in enumerate(grid):
        print(idx, ':', conf)
        train_err, _ = run(conf, args, tracker)
        assert(len(train_err) == args.epochs)


if __name__ == '__main__':
    conf_path = 'configs.json'
    args = get_arguments()
    grid = read_grid(conf_path, args.config[0])
    run_grid(grid, args)
