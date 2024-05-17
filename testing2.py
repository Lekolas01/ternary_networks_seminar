import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
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
    feature_names=iris.feature_names,
    class_names=iris.target_names,
)
plt.show()
