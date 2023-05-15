from pathlib import Path
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import os


class FileDataset(Dataset):
    """
    A generic Dataset wrapper around a dataset where all instances lie in one file.
    The features can be either numerical or categorical.
    """
    def prepare_df(self, df: pd.DataFrame, target: str):
        raise NotImplementedError()
    
    def __init__(self, path: str | Path, range: tuple[float, float]=(0, 1), target='target', normalize=False):
        """
        path: str
            Relative path to the file that contains the dataset.
            - Path must exist and point to a csv file.
            - The first line of the csv file must contain the column names.
        range: (float, float)
            What part of the dataset one wants to access.
            - Both values must be between 0 and 1 and the first must be smaller than the second.
        target: str
            The name of the target variable.
            - Must be one of the column names in the dataset.
        normalize: bool
            Whether or not to normalize the numerical columns to mean = 0 and std = 1.
        """
        assert os.path.isfile(path), f"Path must point to an existing file. Instead got {path}."
        assert len(range) == 2, f"range must be a tuple of length 2. Instead got {range}."
        assert 0 <= range[0] <= range[1] <= 1, f"Invalid range values: {range}"

        df = pd.read_csv(path, skipinitialspace=True)
        assert target in df.columns, f"Target column '{target}' must exist in column names {df.columns}."

        df = df[(df != '?').all(axis=1)] # remove rows with missing values
        for column in df.columns:
            if (normalize and df[column].dtype in ['float64', 'int64']): 
                # normalize numerical columns to mean = 0 and std = 1
                mean = df[column].mean()
                std = df[column].std()
                df[column] = (df[column] - mean) / std
            elif (df[column].dtype == 'object' and column != target):
                # one-hot encode categorical columns
                df = pd.concat([df, pd.get_dummies(df[column], prefix=column, drop_first=False)], axis=1)
                df.drop([column], axis=1, inplace=True)

        # dumme encode target variable and move it to the far right
        df = pd.concat([df, pd.get_dummies(df[target], prefix=target, drop_first=True)], axis=1)
        df.drop([target], axis=1, inplace=True)
        n_rows = len(df)
        ix_low, ix_high = int(range[0] * n_rows), int(range[1] * n_rows)
        
        self.x = torch.tensor(df.iloc[ix_low:ix_high, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[ix_low:ix_high,  -1].values, dtype=torch.float32)


    def __len__(self):
        return len(self.y)
        

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def DataloaderFactory(ds: str, **dl_args):
    datasets = ['adult', 'logical_AND']
    assert ds in datasets, f'DataLoaderFactory does not support the dataset with the name {ds}.'
    dataloaders = []
    
    if (ds == 'mnist'):
        raise NotImplementedError()
        transform = transforms.Compose([transforms.Resize((32, 32)),
            transforms.ToTensor()])
        dataset = datasets.MNIST(root='data/', 
            train=train, 
            transform=transform,
            download=True)
        dataloaders.append(DataLoader(dataset, **dl_args))
        
    elif (ds == 'adult'):
        path = Path('data', ds, 'adult.csv')
        #for filename in ['adult.data', 'adult.names', 'adult.test', 'adult.csv']:
        #    assert filename in os.listdir(path), f'File {filename} not found in {path}.'
        split = 2/3
        dataloaders.append(DataLoader(dataset=FileDataset(path=str(path), range=(0, split)), **dl_args))
        dataloaders.append(DataLoader(dataset=FileDataset(path=path / 'adult.csv', range=(split, 1)), **dl_args))
    elif (ds == 'mushroom'):
        raise NotImplementedError()
        path = Path('data', 'mushroom', 'agaricus-lepiota.data')
        names = ['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
            'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
            'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 
            'spore-print-color', 'population', 'habitat']
        split = 1.0 if train else 0.0
        dataset = FileDataset(path=path, train=train, train_test_split=split, first_is_target=True, names=names)
        dataloaders.append(DataLoader(dataset=dataset, **dl_args))
    elif (ds == 'logical_AND'):
        path = Path('data', 'generated', ds, 'data.csv')
        dataloaders.append(DataLoader(dataset=FileDataset(path=path)))
        dataloaders.append(DataLoader(dataset=FileDataset(path=path)))

    else: raise ValueError('Non-existing dataset: {d}'.format(d=ds))
    
    return dataloaders[0], dataloaders[1]


if __name__ == '__main__':
    train_dl, test_dl = DataloaderFactory(ds='adult', train=True, batch_size=64, shuffle=False)
    print(f"len(dl): {len(train_dl)}")
    print(f"len(dl.dataset): {len(train_dl.dataset)}")
    x, y = next(iter(train_dl))
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")
    
    