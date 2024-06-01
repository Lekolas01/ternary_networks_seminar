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
from ckmeans_1d_dp import ckmeans
from genericpath import isfile
from pandas import DataFrame, Series
from sklearn import tree
from sklearn.exceptions import NotFittedError
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    cross_val_score,
    RandomizedSearchCV,
)
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
    seed = 1
    set_seed(seed)
    # Generate dataframe for parity

    os.makedirs(f_models, exist_ok=True)
    if not os.path.isfile(f_data):
        full_df = get_df(key)
        ds = FileDataset(full_df)
        ds.df.to_csv(f_data, index=False)
    full_df = pd.read_csv(f_data)
    in_shape = full_df.shape[1] - 1  # -1 for target column
    X = full_df.iloc[:, :-1]
    y = full_df.iloc[:, -1]

    # Do a single NN training run on this dataset
    max_epochs = 5000
    bs = 64
    wd = 0.0

    ks = [8]
    lrs = [1e-2, 1e-3]
    l1s = [0.0, 1e-5, 3e-3]
    n_layers = [1, 2]
    runs = DataFrame(
        columns=["idx", "seed", "lr", "k", "n_layer", "l1", "epochs", "wd"]
        + ["train_loss", "nn_acc", "bg_acc", "fidelity", "complexity"]
    )
    all_metrics = []
    n_runs = len(l1s) * len(lrs) * len(n_layers)
    bgs = []
    param_grid = {
        "k": [5, 10],
        "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "l1": [0.0, 1e-5, 1e-4, 3e-3],
        "n_layer": [1, 2, 3],
    }
    re_clf = RuleExtractionClassifier(100, -2, -5, 3, 5000, 0.0)
    grid_search = RandomizedSearchCV(
        re_clf, param_grid, n_iter=10, n_jobs=-1, scoring="f1_macro"
    )
    grid_search.fit(X, y)

    exit()

    for idx, (lr, k, n_layer, l1) in enumerate(
        itertools.product(lrs, ks, n_layers, l1s)
    ):
        print(
            f"{idx = }\t{max_epochs = }\t|{bs = }\t|{wd = }\t|{k = }\t{l1 = }\t|{lr = }\t|{n_layer = }"
        )
        re_clf = RuleExtractionClassifier(lr, k, n_layer, l1, max_epochs, wd)
        re_scores = cross_val_score(re_clf, X, y, cv=5, scoring="f1_macro")
        print(f"{re_scores = }")

        print(f"")
        print("------------------ CART classifier ------------------")

        # Train classifier
        model_path = f"{f_models}/{idx}.pth"

        # if os.path.isfile(model_path) and not new:
        #     model = torch.load(model_path)
        #     print("Model already trained. Skipping this training run...")
        # else:
        spec: NNSpec = [(k - i, k - i - 1, Activation.TANH) for i in range(n_layer)]
        spec.append(((k - n_layer, 1, Activation.SIGMOID)))
        layer1 = spec.pop(0)
        spec.insert(0, (in_shape, k - 1, Activation.TANH))

        model = ModelFactory.get_model_by_spec(spec)
        # train a neural net
        metrics, dl, _ = train_mlp(full_df, model, seed, bs, lr, max_epochs, l1, wd)

        # append losses to losses.csv
        metrics.insert(loc=0, column="idx", value=idx)
        metrics.insert(
            loc=1,
            column="epoch",
            value=[i + 1 for i in range(len(metrics["train_loss"]))],
        )
        all_metrics.append(metrics)
        metrics.to_csv(f_losses, mode="a", index=False, header=(idx == 0))

        full_dl = DataLoader(
            FileDataset(full_df), batch_size=full_df.shape[0], shuffle=False
        )
        nn_acc = accuracy(model, full_dl, "cpu")
        # print(f"full precision model acc: {full_precision_acc}")
        q_model = copy.deepcopy(model)
        while any(isinstance(l, nn.Linear) for l in q_model):
            q_model = quantize_first_lin_layer(q_model, full_df)
        quantized_acc = accuracy(q_model, full_dl, "cpu")
        # print(f"Quantized model acc: {quantized_acc}")
        q_ng, bg, ng_data = nn_to_rule_set(model, full_df)
        bgs.append(bg)
        q_ng_pred = q_ng(ng_data)
        y = np.array(full_df["target"])
        y_int = np.array(y, dtype=int)
        q_ng_acc = np.mean(q_ng_pred == y_int)
        bg_data = {key: np.array(value, dtype=bool) for key, value in ng_data.items()}
        bg_pred = np.array(bg(bg_data), dtype=int)
        bg_acc = np.mean(bg_pred == y_int)
        print(f"Rule Set Accuracy: {bg_acc}")
        X, y = next(iter(full_dl))
        nn_out = model(X).detach().numpy().round().astype(int)
        fidelity = np.mean(nn_out == bg_pred)

        torch.save(model, f"{f_models}/full_precision_{idx}.pth")
        torch.save(q_model, model_path)

        cols = (
            [idx, seed, lr, k, n_layer, l1, metrics.shape[0], wd]
            + [metrics["train_loss"][metrics.shape[0] - 1], nn_acc]
            + [bg_acc, fidelity, bg.complexity()]
        )
        runs.loc[idx] = [str(val) for val in cols]
        runs.to_csv(f_runs, mode="w", header=True, index=False)
        print()
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
