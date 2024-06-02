import copy
import itertools
import os
from argparse import ArgumentParser, Namespace

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import wittgenstein as lw
from ckmeans_1d_dp import ckmeans
from genericpath import isfile
from pandas import DataFrame, Series
from sklearn import tree
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import Tensor
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader

from bool_formula import Activation, overlap
from datasets import FileDataset, get_df
from generate_parity_dataset import parity_df
from models.model_collection import ModelFactory, NNSpec
from q_neuron import QNG_from_QNN, QuantizedLayer
from rule_extraction import nn_to_rule_set
from rule_extraction_classifier import RuleExtractionClassifier
from rule_set import RuleSetGraph
from utilities import accuracy, set_seed

# Grid Search over NN training hyperparameters
#   call train_mlp.py
# For each trained model, extract rule set
# save all rule sets
# create graph for rule sets: dimensions are complexity and accuracy


def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(15, 10))
    tree.plot_tree(
        clf_object,
        filled=True,
        feature_names=feature_names,
        class_names=class_names,
        rounded=True,
    )
    plt.show()


def training_runs(key, f_root, f_data, f_models, f_runs, f_losses):
    seed = 0
    set_seed(seed)
    # Generate dataframe for parity

    os.makedirs(f_models, exist_ok=True)
    full_df, n_targets = get_df(key)
    ds = FileDataset(full_df)
    ds.df.to_csv(f_data, index=False)
    full_df = pd.read_csv(f_data)

    in_shape = full_df.shape[1] - 1  # -1 for target column
    X = full_df.iloc[:, :-1]
    y = full_df.iloc[:, -1]

    # Do a single NN training run on this dataset
    grid_search_epochs = 300
    grid_search_delay = 200
    max_epochs = 8000
    delay = 450
    bs = 64
    wd = 0.0

    l1s = [0.0]
    lrs = [3e-4, 3e-3]
    n_layers = [1, 2]
    runs = DataFrame(
        columns=["idx", "seed", "lr", "k", "n_layer", "l1", "epochs", "wd"]
        + ["train_loss", "nn_acc", "bg_acc", "fidelity", "complexity"]
    )
    param_grid = {
        "k": [8],
        "lr": [3e-4, 3e-3, 1e-2],
        "l1": [1e-5, 1e-3],
        "n_layer": [1, 2, 3],
        "steepness": [2, 4, 8],
    }

    # calculate ripper cross validation
    ripper_clf = lw.RIPPER()
    ripper_clf.fit(X, y)
    ripper_cv_scores = cross_val_score(ripper_clf, X, y, cv=10, scoring="f1_macro")
    print(f"{ripper_cv_scores = }")

    # calculate best hyperparameters for rule extraction
    re_clf = RuleExtractionClassifier(
        0.003, 8, 2, 0.0, grid_search_epochs, 0.0, steepness=8, delay=grid_search_delay
    )
    grid_search = RandomizedSearchCV(
        re_clf, param_grid, n_iter=5, scoring="f1_macro", error_score="raise", verbose=1
    )
    grid_search.fit(X, y)
    p = grid_search.best_params_
    print(f"{p = }")
    re_clf = RuleExtractionClassifier(
        p["lr"],
        p["k"],
        p["n_layer"],
        p["l1"],
        max_epochs,
        0.0,
        steepness=p["steepness"],
        delay=delay,
    )
    re_cv_scores = cross_val_score(re_clf, X, y, cv=10, scoring="f1_macro")
    print(f"{re_cv_scores = }")
    return runs


def main(key: str, retrain=False):
    retrain = False
    f_root = f"runs/{key}"
    f_data = f"{f_root}/data.csv"
    f_models = f"{f_root}/models"
    f_runs = f"{f_root}/runs.csv"
    f_losses = f"{f_root}/losses.csv"

    if not os.path.isfile(f_runs) or retrain:
        runs = training_runs(key, f_root, f_data, f_models, f_runs, f_losses)
        runs.to_csv()
    else:
        runs = pd.read_csv(f_runs)

    n_runs = runs.shape[0]
    models = [torch.load(f"{f_models}/{idx}.pth") for idx in range(n_runs)]
    df = pd.read_csv(f_data)

    n_runs = runs.shape[0]

    complexities = runs["complexity"]
    accs = runs["bg_acc"]
    bgs = [nn_to_rule_set(model, df)[1] for model in models]

    # find "best" models in terms of complexity and validation accuracy
    stats = zip([i for i in range(len(accs))], accs, complexities)
    stats = sorted(stats, key=lambda x: (-x[1], x[2]))
    best_results = stats[0]
    best_model = bgs[best_results[0]]
    print(f"Best Model: {best_model}")
    print(f"Best Model Accuracy = {best_results[1]}")
    print(f"Best Model complexity = {best_results[2]}")
    print(f"Mean accuracy = {np.mean(accs)}")
    print(f"Std. accuracy = {np.std(accs)}")
    print(f"Mean compl. = {np.mean(complexities)}")
    print(f"Std compl. = {np.std(complexities)}")

    sns.scatterplot(x=complexities, y=accs)
    plt.draw()

    df = pd.read_csv(f_data)

    df_metrics = pd.read_csv(f_losses)
    for i in range(n_runs):
        run_info = runs.iloc[i]
        temp = df_metrics[df_metrics["idx"] == i]
        title = f"l1: {run_info['l1']}   |   layers: {run_info['n_layer']}   |   lr: {run_info['lr']}"
        sns.lineplot(x=[i + 1 for i in range(temp.shape[0])], y=temp["train_loss"])
        plt.title(title)
        # plt.show()


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Train an MLP on a binary classification task with an ADAM optimizer."
    )
    parser.add_argument("dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args.dataset)
