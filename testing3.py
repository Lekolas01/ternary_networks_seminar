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

from inspect import Parameter

import torch

# %%
# First load the copy of the Iris dataset shipped with scikit-learn:
from sklearn.datasets import load_iris

from models.model_collection import ModelFactory, SteepTanh
from q_neuron import Perceptron, QuantizedNeuronGraph
from rule_set import QuantizedLayer, RuleSetGraph, RuleSetNeuron

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
print(a.bias)

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

from models.model_collection import SteepTanh

a = SteepTanh(1)
x = np.arange(-5, 5, 0.01)
y = a(torch.tensor(x))
sns.lineplot(x=x, y=y)

plt.show()

# %%
np.arctanh(0.08)
import time

# %%
import timeit

start = timeit.timeit()
time.sleep(1.45)
end = timeit.timeit()
print(end - start)
# %%
from q_neuron import Perceptron, QuantizedNeuronGraph
from rule_set import RuleSetNeuron

q_neuron = Perceptron(
    "y", {"x1": 3.0, "x2": 3.0, "x3": 1.0, "x4": 1.0, "x5": 1.0}, -5.5
)
RuleSetGraph.from_QNG
rs = RuleSetNeuron(q_neuron, QuantizedNeuronGraph([q_neuron]), True)
print(rs)

# %%
a = SteepTanh(10)
print(a.state_dict())
b = torch.tensor(3)
print(b)
print(b.shape)
m = nn.Linear()
m = torch.nn.utils.skip_init()


# %%
import torch.nn as nn


def debug_input(m, inputs):
    # Allows for examination and modification of the input before the forward pass.
    # Note that inputs are always wrapped in a tuple.
    print(m.weight)
    print(m.weight.shape)
    return inputs[0]


def forward_hook(m, inputs, output):
    # Allows for examination of inputs / outputs and modification of the outputs
    # after the forward pass. Note that inputs are always wrapped in a tuple while outputs
    # are passed as-is.
    print(f"{output = }")
    ans = torch.sign(output.detach())
    print(f"{output = }")
    print(f"{ans = }")
    return ans


class Temp(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.l1 = nn.Linear(in_features, out_features)
        self.act = nn.Tanh()
        self.l1.register_forward_pre_hook(debug_input)
        self.act_handle = self.act.register_forward_hook(forward_hook)

    def forward(self, x):
        return self.act(self.l1(x))


N = 10
D = 3

x = 2 * torch.rand((N, D)) - 1
y = 2 * torch.rand((D, 3)) - 1
a = nn.Linear(10, 20)
print(f"{a.weight = }")
print(f"{a.weight.t() = }")
print(f"{a.bias = }")
print(f"{a.bias.t() = }")
m = Temp(D, 1)
# print(f"{x.shape = }")
# print(f"{x = }")
# print(f"{m(x) = }")
# forward_hook_handle = m.register_forward_hook(forward_hook)

# %%
a = {"a": 0.5, "b": 0.3}
c = list(a.items())
print(c)
c.sort(key=lambda x: x[1])
print(c)
