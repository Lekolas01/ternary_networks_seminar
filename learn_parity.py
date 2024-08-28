import os
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wittgenstein as lw
from pandas import DataFrame, Series
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.utils import shuffle

from datasets import FileDataset, get_df
from rule_extraction import nn_to_rule_set
from rule_extraction_classifier import RuleExtractionClassifier
from utilities import set_seed


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


def load_dataset(key: str) -> tuple[DataFrame, Series]:
    f_data = f"data/{key}.csv"
    if not os.path.isfile(f_data):
        print(f"Loading dataset with key {key}")
        df = get_df(key)
        ds = FileDataset(df, encode=True)
        ds.df.to_csv(f_data, index=False)
        print(f"Saved dataset to {f_data}")

    df = pd.read_csv(f_data)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def training_runs(key):
    seed = 1
    set_seed(seed)

    X, y = load_dataset(key)

    max_epochs = 8000

    runs = DataFrame(
        columns=["idx", "seed", "lr", "layer_width", "n_layer", "l1", "epochs", "wd"]
        + ["train_loss", "nn_acc", "bg_acc", "fidelity", "complexity"]
    )

    re_clf = RuleExtractionClassifier(
        lr=0.003,
        layer_width=3,
        n_layer=1,
        l1=3e-5,
        epochs=max_epochs,
        wd=0.0,
        steepness=6,
        delay=400,
    )

    # find best hyperparameter settings
    # abcdefg: Best parameters: {'l1': 1e-05, 'layer_width': 8, 'lr': 0.0003, 'n_layer': 1, 'steepness': 6}
    param_grid = {
        "layer_width": [8],
        "lr": [3e-4, 3e-3],
        "l1": [1e-5],
        "n_layer": [1, 2],
        "steepness": [2, 6],
    }

    random_search = RandomizedSearchCV(
        re_clf,
        param_grid,
        scoring="accuracy",
        n_iter=5,
        cv=5,
        error_score="raise",
    )

    # random_search.fit(X, y)
    # re_clf.set_params(**random_search.best_params_)

    # with these new-found hyperparameter settings, we do one final cross validation
    # re_cv_scores = cross_val_score(
    #     re_clf, X, y, cv=5, scoring="accuracy", error_score="raise"
    # )
    # print(f"cross validation scores: {re_cv_scores}")

    # Best parameters:
    # abcdefg:
    re_clf.set_params(l1=1e-03, layer_width=8, lr=0.003, n_layer=2, steepness=6)
    print(re_clf.get_params())
    print(f"Best parameters: {re_clf.get_params()}")

    # finally, we do a single fit on the whole dataset, and observe the learned rule set
    re_clf.fit(X, y)
    print(f"Rule set: {str(re_clf.bool_graph)}")
    print(f"{re_clf.q_ng = }")
    print(f"{re_clf.bool_graph.complexity() = }")
    return runs


def main(key: str, retrain=False):
    retrain = False
    f_root = f"runs/{key}"
    f_data = f"{f_root}/data.csv"
    f_models = f"{f_root}/models"
    f_runs = f"{f_root}/runs.csv"
    f_losses = f"{f_root}/losses.csv"

    if not os.path.isfile(f_runs) or retrain:
        runs = training_runs(key)
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
