from ucimlrepo import fetch_ucirepo

import datasets

# fetch dataset
mushroom = fetch_ucirepo(id=73)

# data (as pandas dataframes)
X = mushroom.data.features
y = mushroom.data.targets

# metadata
print(mushroom.metadata)

# variable information
train_ds, test_ds = datasets.get_dataset("mushroom")
print(train_ds[0:5])
print(X.head())
print(len(train_ds[0]))
print(len(train_ds))
