import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

from bool_formula import AND, NOT, Bool, Literal
from neuron import possible_data, powerset


def gen_data(func: Bool, n: int = 0, seed: int | None = None) -> pd.DataFrame:
    """Generate data from a boolean/logical function with k variables and save it in a DataFrame.
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
        randomly from the whole input space.
    seed: int | None
        Set the seed for the random sampling. Obviously only relevant if n >= 1.

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

    if n <= 0:
        # generate a data point for each possible point in the input space
        for datapoint in possible_data(vars):
            df.loc[len(df)] = func(datapoint)  # type: ignore
    elif n >= 1:
        # generate n randomly selected data points from the input space
        random.seed(seed)
        for _ in range(n):
            interpretation = {l: random.random() >= 0.5 for l in vars}
            interpretation[target_col] = func(interpretation)
            df.loc[len(df)] = interpretation  # type: ignore
    return df.astype(int)


def main(path, n_rows, n_vars, sep=","):
    assert os.access(path, os.W_OK), f"path {path} must be writable."
    assert (
        isinstance(n_rows, int) and n_rows >= 1
    ), f"n_rows must be int type and greater than 0."
    assert (
        isinstance(n_vars, int) and n_vars >= 1
    ), f"n_vrs must be int type and greater than 0."

    vars = [f"x{i}" for i in range(n_vars)]  # create variable names
    vars.append("target")

    n_cols = n_vars + 1
    thresholds = [n * n_rows for n in [0.15, 0.35, 0.6, 1.0]]
    code = 0
    with open(path, "w") as of:
        # header line
        of.writelines(f"{sep.join(vars)}\n")

        for i in range(n_rows):
            if i >= thresholds[0]:
                thresholds.pop(0)
                code += 1
            row = (
                np.repeat("1", (n_cols))
                if len(thresholds) == 1
                else np.array(
                    [str(int((code >> j) % 2 == 1)) for j in range(n_vars - 1, -1, -1)]
                    + [0]
                )
            )
            print(row)
            of.write(f"{sep.join(row)}\n")


if __name__ == "__main__":
    """
    This script creates a dataset following a logical formula.
    """
    formula_name = "logical_AND"
    dir_path = Path("./data", "generated", formula_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, "data.csv")
    n_rows, n_vars = 100, 2
    data = gen_data(AND("x1", NOT(Literal("x2"))), 100)
