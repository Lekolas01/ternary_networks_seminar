import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd

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
        "--path",
        default="./",
        help="Directory path for the new dataset file, starting from root folder",
    )
    parser.add_argument(
        "--no_header",
        action="store_true",
        help="If specified, will include the generated header in the target file.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="If specified, the datasamples will be shuffled.",
    )
    args = parser.parse_args()
    return args


def gen_parity_data(path: str, k: int, shuffle: bool, n=0) -> pd.DataFrame:
    keys = [f"x{i + 1}" for i in range(k)]
    parity = PARITY(keys)
    df = gen_data(parity, col_order=keys, shuffle=shuffle, n=n)
    file_path = Path(path, f"parity{k}").with_suffix(".csv")
    if os.path.isfile(file_path):
        print(
            f"File at path {file_path} already exists. Do you want to overwrite it[y/n]?"
        )
        val = input()

    if not os.path.isfile(file_path) or val in ["y", "Y"]:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"Saved parity dataset to {file_path}.")
    else:
        print("Cancelling operation...")
    return df


if __name__ == "__main__":
    args = get_arguments()
    gen_parity_data(args.path, args.k, args.shuffle, args.n)
