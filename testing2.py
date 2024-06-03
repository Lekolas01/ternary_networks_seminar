import time
import timeit
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from torch.utils.data import DataLoader
from ucimlrepo import fetch_ucirepo

from datasets import FileDataset

start = timer()
time.sleep(1.45)
end = timer()
print(end - start)  # Time in seconds, e.g. 5.38091952400282
exit()

# fetch dataset
car_evaluation = fetch_ucirepo(id=19)

# data (as pandas dataframes)
X = car_evaluation.data.features
y = car_evaluation.data.targets

# dl = DataLoader(FileDataset(data), batch_size=batch_size, shuffle=True)

cX_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a decision tree classifier
clf = DecisionTreeClassifier(max_depth=3)

# Train the model on the training data
classifier = clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

plot_tree(
    clf,
    filled=True,
    label="all",
    rounded=True,
    impurity=False,
    feature_names=car_evaluation.feature_names,
    class_names=car_evaluation.target_names,
)
plt.show()
