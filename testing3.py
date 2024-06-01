"""
=======================================================================
Plot the decision surface of decision trees trained on the iris dataset
=======================================================================

Plot the decision surface of a decision tree trained on pairs
of features of the iris dataset.

See :ref:`decision tree <tree>` for more information on the estimator.

For each pair of iris features, the decision tree learns decision
boundaries made of combinations of simple thresholding rules inferred from
the training samples.

We also show the tree structure of a model built on all of the features.
"""

import torch

# %%
# First load the copy of the Iris dataset shipped with scikit-learn:
from sklearn.datasets import load_iris

from models.model_collection import ModelFactory, SteepTanh
from rule_set import QuantizedLayer

iris = load_iris()


# %%
# Display the decision functions of trees trained on all pairs of features.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02


for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]],
    )

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

plt.suptitle("Decision surface of decision trees trained on pairs of features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")

# %%
# Display the structure of a single decision tree trained on all the features
# together.
from sklearn.tree import plot_tree

plt.figure()
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf, filled=True)
plt.title("Decision tree trained on all the iris features")
plt.show()

# %%
a = [1, 2, 3, 4, 5, 6, 7]
l = len(a)
a[-len(a)] <= a[-1]

# %%
import torch

a = torch.normal(0, 1, size=(5, 2))
w = torch.rand((2, 1))
temp = a @ w
print(temp)
temp = torch.normal(0, 1, size=(1024, 8))
y_low = -abs(torch.rand(8))
y_high = abs(torch.rand(8))
torch.where(temp >= 0, y_high, y_low).shape


# %%
import torch.nn as nn

a = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), nn.Linear(4, 3), nn.Sigmoid())
type(a[0])
import torch

# %%
from rule_set import QuantizedLayer

ql = QuantizedLayer(
    torch.tensor([[1.0, 2], [4, 6], [-2.5, 1.2]]),
    torch.tensor([-2.0, -1.0]),
    torch.tensor([-0.64, -0.91]),
    torch.tensor([0.96, 0.44]),
)
x = torch.rand((5, 3))
print(f"{ql.weight = }")
print(f"{ql.bias = }")
print(f"{x = }")
print(f"{ql(x) = }")

# %%
import torch.nn as nn

a = nn.Sequential()
x = torch.rand((5, 3))
print(x)
print(a(x))

# %%
a = nn.Linear(in_features=5, out_features=8)
print(a.weight.requires_grad_(False))
print(a.bias.requires_grad_(False))
print(a.weight.shape)
a.bias += 1
print(a.bias)

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from models.model_collection import SteepTanh

a = SteepTanh(1)
x = np.arange(-5, 5, 0.01)
y = a(torch.tensor(x))
sns.lineplot(x=x, y=y)

plt.show()

# %%
np.arctanh(0.08)
# %%
