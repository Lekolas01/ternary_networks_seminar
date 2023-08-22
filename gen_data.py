import numpy as np
from pathlib import Path
import os
from bool_formula import *
import pandas as pd
from typing import Optional


def generate_data(
    n_samples: int, func: Boolean, vars: Optional[list[str]] = None
) -> pd.DataFrame:
    assert (
        isinstance(n_samples, int) and n_samples >= 1
    ), f"n_rows must be int type and greater than 0."
    if not vars:
        vars = list(func.all_literals())

    columns = vars.copy()

    target_col = "target"
    columns.append(target_col)
    df = pd.DataFrame(columns=columns)
    # for every sample, pick a random interpretation,
    # calculate the value and add that row to the DataFrame
    for _ in range(n_samples):
        interpretation = random_interpretation(set(vars))
        interpretation[target_col] = func(interpretation)
        df.loc[len(df)] = interpretation  # type: ignore
    return df


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
    data = np.empty((n_rows, n_cols), dtype=bool)
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
    data = generate_data(100, AND([Literal("x1"), Literal("x2", False)]))
