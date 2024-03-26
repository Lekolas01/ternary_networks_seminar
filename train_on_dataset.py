# this script runs a grid search on an MLP training proceure, including conversion to if then rules.
# and it saves all the models
from argparse import ArgumentParser, Namespace

from config import read_grid
from datasets import get_dataset


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Generate and save a dataset following a boolean function. The dataset contains every point from the possible input space exactly once."
    )
    parser.add_argument("dataset", help="Relative path to the categorical dataset.")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    dataset = args.dataset
    ds_path = "datasets/" + dataset + ".csv"
    conf_path = "configs.json"
    grid = read_grid(conf_path, dataset)
    print(grid)


if __name__ == "__main__":
    main()
