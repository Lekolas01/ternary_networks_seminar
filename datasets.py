import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from ucimlrepo import fetch_ucirepo

from bool_parse import ExpressionEvaluator
from gen_data import gen_data
from generate_parity_dataset import parity_df


class FileDataset(Dataset):
    """
    A generic Dataset wrapper around a dataset where all instances lie in one file.
    The features can be either numerical or categorical.
    """

    def prepare_df(self, df: pd.DataFrame, target: str):
        raise NotImplementedError()

    def __init__(
        self,
        df: pd.DataFrame,
        range: tuple[float, float] = (0, 1),
        target="target",
        normalize=False,
    ):
        """
        path: str
            Relative path to the file that contains the dataset.
            - Path must exist and point to a csv file.
            - The first line of the csv file MUST contain the column names.
        range: (float, float)
            What part of the dataset one wants to access.
            - Both values must be between 0 and 1 and the first must be smaller than the second.
        target: str
            The name of the target variable.
            - Must be one of the column names in the dataset.
        normalize: bool
            Whether or not to normalize the numerical columns to mean = 0 and std = 1.
        """
        assert (
            target in df.columns
        ), f"Target column '{target}' must exist in column names {df.columns}."
        # df = copy.deepcopy(df)
        df = df[(df != "?").all(axis=1)]  # remove rows with missing values
        for column in df.columns:
            if normalize and df[column].dtype in ["float64", "int64"]:
                # normalize numerical columns to mean = 0 and std = 1
                mean = df[column].mean()
                std = df[column].std()
                df[column] = (df[column] - mean) / std

            elif df[column].dtype == "object" and column != target:
                # one-hot encode categorical columns
                df = pd.concat(
                    [df, pd.get_dummies(df[column], prefix=column, drop_first=False)],
                    axis=1,
                )
                df.drop([column], axis=1, inplace=True)

        self.n_target = 0
        if len(df[target].unique()) > 2:
            dummies = [df, pd.get_dummies(df[target], prefix=target, drop_first=False)]
            df = pd.concat(
                dummies,
                axis=1,
            )
            df.drop([target], axis=1, inplace=True)
            self.n_target = len(dummies)
        elif len(df[target].unique()) == 2:
            df = pd.concat(
                [df, pd.get_dummies(df[target], prefix=target, drop_first=True)],
                axis=1,
            )
            df.drop([target], axis=1, inplace=True)
            df.rename(columns={df.columns[-1]: "target"}, inplace=True)
            self.n_target = 1
        else:
            raise ValueError("Must have more than one constant target value.")

        self.df = df
        n_rows = len(df)
        ix_low, ix_high = int(range[0] * n_rows), int(range[1] * n_rows)

        self.x = torch.tensor(
            df.iloc[ix_low:ix_high, : -self.n_target].values, dtype=torch.float32
        )
        self.y = torch.tensor(
            df.iloc[ix_low:ix_high, -self.n_target].values, dtype=torch.float32
        )
        self.shape = (len(self.y), self.x.shape[1] + 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_dataset(ds: str) -> tuple[FileDataset, FileDataset]:
    """
    For obtaining two FileDatasets (one train, one validation) based on dataset name.
    """
    datasets = []
    match ds:
        case "adult":
            path = Path("data/adult/adult.csv")
            split = 2 / 3
            datasets.append(
                FileDataset(path=str(path), range=(0, split), normalize=True)
            )
            datasets.append(
                FileDataset(path=str(path), range=(split, 1), normalize=True)
            )

        case "mushroom":
            from ucimlrepo import fetch_ucirepo

            # fetch dataset
            mushroom = fetch_ucirepo(id=73)
            # datasets.append(FileDataset(mushroom.data))
            path = Path("data", "mushroom", "agaricus-lepiota.data")
            names = [
                "edible",
                "cap-shape",
                "cap-surface",
                "cap-color",
                "bruises",
                "odor",
                "gill-attachment",
                "gill-spacing",
                "gill-size",
                "gill-color",
                "stalk-shape",
                "stalk-root",
                "stalk-surface-above-ring",
                "stalk-surface-below-ring",
                "stalk-color-above-ring",
                "stalk-color-below-ring",
                "veil-type",
                "veil-color",
                "ring-number",
                "ring-type",
                "spore-print-color",
                "population",
                "habitat",
            ]
            datasets.append(FileDataset(path=path, target="edible"))
            datasets.append(FileDataset(path=path, target="edible"))
        case "logical_AND":
            path = Path("data", "generated", ds, "data.csv")
            datasets.append(FileDataset(path=path))
            datasets.append(FileDataset(path=path))
        case _:
            raise ValueError("Non-existing dataset: {d}".format(d=ds))
    return datasets[0], datasets[1]


def get_df(key: str) -> pd.DataFrame:
    def get_df_from_uci(id: int) -> pd.DataFrame:
        temp = fetch_ucirepo(id=id)
        X = temp.data.features  # type: ignore
        y = temp.data.targets  # type: ignore
        ans = X.join(y)
        ans.rename(columns={y.columns[0]: "target"}, inplace=True)
        return ans

    match key:
        case "parity10":
            return parity_df(k=10, shuffle=False, n=1024)
        case "adult":
            return get_df_from_uci(2)
        case "mushroom":
            return get_df_from_uci(73)
        case "car_evaluation":
            ans = get_df_from_uci(19)
            return ans
        case "abcdefg":
            e = ExpressionEvaluator()
            fn = e.parse("(a | b) & (c | d) & (e | (f & g))")
            return gen_data(fn, dead_cols=3, shuffle=True, n=1024)
        case _:
            raise ValueError


if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset(ds="adult")
    print(f"{len(train_dataset) = }")
    x, y = next(iter(train_dataset))
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")
    print(f"{x = }")
    print(f"{y = }")
    print(f"{train_dataset[7] = }")
