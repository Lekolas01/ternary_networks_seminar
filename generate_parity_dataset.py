from argparse import ArgumentParser, Namespace
from pathlib import Path

from bool_formula import PARITY
from gen_data import gen_data


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Generate and save a dataset for a k-variable parity function. The dataset contains every point from the possible input space exactly once."
    )
    parser.add_argument(
        "-n",
        type=int,
        default=0,
        help="Number of samples you want to create. If not specified, it will create every sample exactly once.",
    )
    parser.add_argument(
        "k",
        type=int,
        help="What parity function you want.",
    )
    parser.add_argument(
        "--no_header",
        action="store_true",
        help="If specified, will exclude the generated header in the target file.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="If specified, the datasamples will be shuffled.",
    )
    args = parser.parse_args()
    return args


args = get_arguments()
n_vars = args.k
keys = [f"x{i + 1}" for i in range(n_vars)]
parity = PARITY(keys)
df = gen_data(parity, col_order=keys, shuffle=args.shuffle, n=args.n)
dir_path = Path("./data", "generated")
file_path = Path(dir_path, f"parity{n_vars}").with_suffix(".csv")

df.to_csv(file_path, index=False)
print(f"Created a dataset with shape {df.shape}.")
print(f"Saved dataset to {file_path}")
