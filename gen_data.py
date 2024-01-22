import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from bool_formula import AND, NOT, OR, Bool, Literal
from neuron import possible_data, powerset


def gen_data(
    func: Bool, n: int = 0, seed: int | None = None, reverse=False
) -> pd.DataFrame:
    """Generate data from a boolean/logical function with k variables and save it in a pandas DataFrame.
    The target variable is saved in the "target" column of the df.
    The function chooses data points either deterministically or randomly, based on parameter n.

    Parameters
    ----------
    func : Bool
        The target function for which to generate a dataset.
    n : int | None
        If n is <= 0, the DataFrame will contain every possible sample from the input space exactly once,
        i.e. the DataFrame will have exactly 2**n data points.
        If n is an int >= 1, the DataFrame will contain n data points with each data point sampled
        uniform randomly from the whole input space.
    seed: int | None
        Set the seed for the random sampling. Only relevant if n >= 1.

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame that contains the data, with shape (n, k + 1).
        The first k columns are named after each variable occuring in func sorted lexicographically,
        the last column is the "target" column.
    """
    vars = sorted(list(func.all_literals()))
    target_col = "target"
    df = pd.DataFrame(columns=vars + [target_col])
    if reverse:
        vars.reverse()

    if n <= 0:
        # generate a data point for each possible point in the input space
        data = possible_data(vars)
        for datapoint in data:
            df[datapoint] = data[datapoint]
        df[target_col] = func(data)

    elif n >= 1:
        # generate n randomly selected data points from the input space
        random.seed(seed)
        for _ in range(n):
            interpretation = {l: random.random() >= 0.5 for l in vars}
            interpretation[target_col] = func(interpretation)
            df.loc[len(df)] = interpretation  # type: ignore
    return df.astype(int)


if __name__ == "__main__":
    """
    This script creates a dataset following a logical formula.
    """
    formula_name = "abcdefg"
    dir_path = Path("./data", "generated", formula_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, "data.csv")
    f = AND(OR("a", "b"), AND(OR("c", "d"), OR("e", AND("f", "g"))))
    data = gen_data(f, reverse=True)
    print(data)
