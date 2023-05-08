import numpy as np
from pathlib import Path
import random
import os

def main(path, n_rows, n_vars, sep=','):
    assert os.access(path, os.W_OK), f"path {path} must be writable."
    assert isinstance(n_rows, int) and n_rows >= 1, f"n_rows must be int type and greater than 0."
    assert isinstance(n_vars, int) and n_vars >= 1, f"n_vrs must be int type and greater than 0."

    split = 0.2
    vars = [f"x{i}" for i in range(n_vars)] # create variable names
    vars.append('target')
    
    n_cols = n_vars + 1
    data = np.empty((n_rows, n_cols), dtype=bool)
    with open(path, 'w') as of:
        # header line
        of.writelines(f"{sep.join(vars)}\n")
        for i in range(n_rows):
            if random.random() <= split:
                # positive target
                row = np.repeat('1', (n_cols))
            else:
                # negative target
                code = random.randint(0, 2**n_vars-1)
                row = np.array([str(int((code >> j) % 2 == 1)) for j in range(n_vars-1, -1, -1)] + [0])
            print(row)
            of.write(f"{sep.join(row)}\n")


if __name__ == '__main__':
    """
    This script creates a dataset following a logical formula.
    """
    formula_name = 'logical_and'
    path = Path('data', 'generated', formula_name, 'data.csv')
    n_rows, n_vars = 100, 2
    main(path, n_rows, n_vars)

