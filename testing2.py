import itertools
from graphlib import TopologicalSorter

import numpy as np

from bool_formula import PARITY

a = [3, 1, 5, 2]


keys = [f"x{i + 1}" for i in range(5)]
a = PARITY(keys)

data = {key: np.random.binomial(1, 0.5, 5) for key in keys}
print(data)
print(a(data))
