from pathlib import Path

from bool_formula import PARITY
from gen_data import gen_data

n_vars = 5
keys = [f"x{i + 1}" for i in range(n_vars)]
parity = PARITY(keys)
df = gen_data(parity, col_order=keys, n=1000, shuffle=True)
dir_path = Path("./data", "generated")
file_path = Path(dir_path, f"parity{n_vars}").with_suffix(".csv")

df.to_csv(file_path, index=False)
