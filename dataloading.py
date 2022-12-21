from pathlib import Path
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms


class FileDataset(torch.utils.data.Dataset):
    """
    A generic Dataset wrapper around a dataset where all instances lie in one file.
    The features can be both numerical as well as categorical.
    """
    def prepare_df(self, df: pd.DataFrame):
        """ Dummy encode all categorical columns of df. """
        for column in df:
            if (df[column].dtype in ['float64', 'int64']): # numerical
                # normalize column
                mean = df[column].mean()
                std = df[column].std()
                df[column] = (df[column] - mean) / std

        for column in df:
            if (df[column].dtype == 'object'): #i.e. categorical
                df = pd.concat([df, pd.get_dummies(df[column], prefix=column, drop_first=True)], axis=1)
                df.drop([column], axis=1, inplace=True)
        return df


    def __init__(self, root: str, train: bool, train_test_split: float, names: list, target: str):
        """
        train: bool
            Whether to access the train or test set.
        train_test_split: float
            Must be between 0 and 1. ratio of train set to whole dataset.
        target:
            Specifies which of the columns is the target variable.
        """
        assert target in names, 'name of target variable must be in names'
        df = pd.read_csv(root, names=names)
        print(f"len(df.columns): {len(df.columns)}")
        print(f"df.columns: {list(df.columns)}")
        df = self.prepare_df(df)
        print(f"len(df.columns): {len(df.columns)}")
        print(f"df.columns: {list(df.columns)}")
        
        self.df = df
        self.train = train
        self.n_samples = len(self.df)
        self.n_train_samples = int(self.n_samples * train_test_split)
        self.n_test_samples = self.n_samples - self.n_train_samples

        if train:
            x = df.iloc[:self.n_train_samples,1:].values
            y = df.iloc[:self.n_train_samples,0].values
        else:
            x = df.iloc[self.n_train_samples:,1:].values
            y = df.iloc[self.n_train_samples:,0].values

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        print(f"self.x.shape: {self.x.shape}")
        print(f"self.y.shape: {self.y.shape}")
        pass
        

    def __len__(self):
        return len(self.y)
        

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def DataloaderFactory(ds: str, train: bool, **dl_args):
    if (ds == 'mnist'):
        transform = transforms.Compose([transforms.Resize((32, 32)),
            transforms.ToTensor()])
        dataset = datasets.MNIST(root='data/', 
            train=train, 
            transform=transform,
            download=True)
        return DataLoader(dataset, **dl_args)

    elif (ds == 'adult'):
        root = Path('data', 'adult', 'adult.full')
        names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'less_than_50']
        dataset = FileDataset(root=root, train=train, train_test_split=2/3, names=names, target='less_than_50')
        dl = DataLoader(dataset=dataset, **dl_args)
        return dl

    elif (ds == 'mushroom'):
        root = Path('data', 'mushroom', 'agaricus-lepiota.data')
        names = ['edible', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
            'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
            'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 
            'spore-print-color', 'population', 'habitat']
        train_test_split = 1.0 if train else 0.0
        dataset = FileDataset(root=root, train=train, train_test_split=train_test_split, first_is_target=True, names=names)
        return DataLoader(dataset=dataset, **dl_args)

    raise ValueError('Non-existing dataset: {d}'.format(d=ds))
    

if __name__ == '__main__':
    dl = DataloaderFactory(ds='adult', train=True, batch_size=64, shuffle=False)
    print(f"len(dl): {len(dl)}")
    print(f"len(dl.dataset): {len(dl.dataset)}")
    x, y = next(iter(dl))
    print(f"x.shape: {x.shape}")
    print(f"y.shape: {y.shape}")
    
    