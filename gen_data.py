import os
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from bool_formula import Bool
from bool_parse import ExpressionEvaluator
from neuron import possible_data


def gen_data(
    func: Bool, n=0, dead_cols=0, seed: int | None = None, shuffle=False, verbose=False
) -> pd.DataFrame:
    """Generate data from a boolean/logical function with k unqiue variables and save it in a pandas DataFrame.
    The target variable is saved in the "target" column of the df.
    The function chooses data points either deterministically or randomly, based on parameter n.

    Parameters
    ----------
    func : Bool
        The target function for which to generate a dataset.
    n : int | None
        If n is <= 0, the DataFrame will contain every possible sample from the input space exactly once,
        i.e. the DataFrame will contain exactly 2**k data points, where k is the number
        of different variables in func.
        If n is an int >= 1, the DataFrame will contain n data points with each data point sampled
        uniform randomly from the whole input space.
    dead_cols : int
        How many irrelevant columns you want to add to the dataset, that don't have any impact on
        the target function (default 0). This number does not count towards k, i.e. the
        size of the dataset in the first dimension is not impacted by these columns.
        If dead_cols <= 0, ignore it.
    seed: int | None
        Set the seed for the random sampling. Only relevant if n >= 1.
    shuffle: bool
        Whether to have the data samples be in random order. Only relevant if n <= 0.
    verbose: bool
        Whether or not to print additional debuggin information to stdout.

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame that contains the data, with shape (n, k + dead_vars + 1).
        The first k columns are named after each variable occuring in func sorted lexicographically,
        the last column is the "target" column, and the ones in between are the irrelevant columns.
    """
    vars = sorted(list(func.all_literals()))
    target_var = "target"
    dead_vars = [f"dead{i + 1}" for i in range(dead_cols)]
    df = pd.DataFrame(columns=vars + dead_vars + [target_var])
    if verbose:
        print(f"Generating dataset for the following expression: {func}...")
        print(f"Columns: {vars}")
    if n <= 0:
        # generate a data point for each possible point in the input space
        data = possible_data(vars, is_float=False, shuffle=shuffle)
        for datapoint in data:
            df[datapoint] = data[datapoint]
        df[target_var] = func(data)
        n = 2 ** len(vars)

    elif n >= 1:
        # generate n randomly selected data points from the input space
        random.seed(seed)
        for i in range(n):
            interpretation = {l: np.array(random.random() >= 0.5) for l in vars}
            interpretation[target_var] = func(interpretation)
            df.loc[len(df)] = interpretation  # type: ignore
    for dead_var in dead_vars:
        df[dead_var] = np.random.choice([0, 1], n, True, np.array([0.5, 0.5]))
    return df.astype(int)


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Generate and save a dataset following a boolean function. The dataset contains every point from the possible input space exactly once."
    )
    parser.add_argument(
        "target_fn", help="The target function as a boolean expression."
    )
    parser.add_argument(
        "path",
        help="Save location for the new dataset file, starting from ./data/generated.",
    )
    parser.add_argument(
        "--no_header",
        action="store_true",
        help="If specified, will exclude the header in the file.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="If specified, the datasamples will be shuffled.",
    )
    parser.add_argument(
        "--n_dead",
        type=int,
        default=0,
        help="Number of dead variables to add to the dataset.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    This script creates a dataset following a logical formula "target_fn" and saves
    it in a file. See argument parser for more information.
    """
    args = get_arguments()
    e = ExpressionEvaluator()
    f = e.parse(args.target_fn)
    dir_path = Path("./data", "generated")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = Path(dir_path, args.path).with_suffix(".csv")
    data = gen_data(f, dead_cols=args.n_dead, shuffle=args.shuffle, verbose=True)
    print(f"Final data shape: {data.shape}")
    data.to_csv(file_path, index=False, header=not args.no_header)
    print(f"Saved dataset to {file_path}.")
