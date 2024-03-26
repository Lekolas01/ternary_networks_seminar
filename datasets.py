import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class FileDataset(Dataset):
    """
    A generic Dataset wrapper around a dataset where all instances lie in one file.
    The features can be either numerical or categorical.
    """

    def prepare_df(self, df: pd.DataFrame, target: str):
        raise NotImplementedError()

    def __init__(
        self,
        path: str | Path,
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
        assert os.path.isfile(
            path
        ), f"Path must point to an existing file. Instead got {path}."
        assert (
            len(range) == 2
        ), f"range must be a tuple of length 2. Instead got {range}."
        assert 0 <= range[0] <= range[1] <= 1, f"Invalid range values: {range}"

        df = pd.read_csv(path, skipinitialspace=True)
        assert (
            target in df.columns
        ), f"Target column '{target}' must exist in column names {df.columns}."

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

        # dummy encode target variable and move it to the far right
        if len(df[target].unique()) > 1:
            df = pd.concat(
                [df, pd.get_dummies(df[target], prefix=target, drop_first=True)],
                axis=1,
            )
            df.drop([target], axis=1, inplace=True)

        n_rows = len(df)
        ix_low, ix_high = int(range[0] * n_rows), int(range[1] * n_rows)

        self.x = torch.tensor(df.iloc[ix_low:ix_high, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[ix_low:ix_high, -1].values, dtype=torch.float32)

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


if __name__ == "__main__":
    train_dataset, test_dataset = get_dataset(ds="adult")
    print(f"{len(train_dataset) = }")
    x, y = next(iter(train_dataset))
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")
    print(f"{x = }")
    print(f"{y = }")
    print(f"{train_dataset[7] = }")
