import copy
import os
from argparse import ArgumentParser, Namespace
from cProfile import label
from distutils.sysconfig import customize_compiler
from enum import unique
from turtle import right

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wittgenstein as lw
from pandas import DataFrame, Series
from sklearn import tree
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import (
    ParameterSampler,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.utils import shuffle

from datasets import FileDataset, get_df
from rule_extraction import nn_to_rule_set
from rule_extraction_classifier import RuleExtractionClassifier
from rule_set import RuleSetGraph
from utilities import accuracy, set_seed


def custom_score(acc, complexity):
    l = 0.05
    return 1 / ((1.0 + l - acc) * complexity)


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
    seed = 0
    set_seed(seed)

    X, y = load_dataset(key)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    runs = DataFrame(
        columns=["idx", "seed", "lr", "layer_width", "n_layer", "l1", "epochs", "wd"]
        + ["train_loss", "nn_acc", "bg_acc", "fidelity", "complexity"]
    )

    re_clf = RuleExtractionClassifier(
        lr=0.003,
        layer_width=3,
        n_layer=1,
        l1=3e-5,
        epochs=5000,
        wd=0.0,
        steepness=6,
        delay=200,
    )

    # find best hyperparameter settings
    # abcdefg: Best parameters: {'l1': 1e-05, 'layer_width': 8, 'lr': 0.0003, 'n_layer': 1, 'steepness': 6}
    param_grid = {
        "layer_width": [5, 10],
        "lr": [3e-4, 3e-3],
        "l1": [1e-4, 3e-3],
        "n_layer": [1, 2, 3],
        "steepness": [2, 4, 8],
    }

    n_iter = 15
    # do a random search over the hyperparameter space
    sampler = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=seed + 1))

    best_model, best_params = RuleSetGraph([]), {}
    max_score = float("-inf")
    accs, complexities, scores = np.zeros(n_iter), np.zeros(n_iter), np.zeros(n_iter)
    fidelities = np.zeros(n_iter)
    for i, sample in enumerate(sampler):
        re_clf.set_params(**sample)
        re_clf.fit(X_train, y_train)
        accs[i] = accuracy_score(y_test, re_clf.predict(X_test))
        complexities[i] = re_clf.bool_graph.complexity()
        scores[i] = custom_score(accs[i], complexities[i])
        fidelities[i] = re_clf.fid_rule_set

        if scores[i] > max_score:
            max_score = scores[i]
            best_model = copy.copy(re_clf.bool_graph)
            best_params = sample

    print(f"{accs = }")
    print(f"{complexities = }")
    print(f"{scores = }")

    f_figures = f"figures/{key}/"
    if not os.path.isdir(f_figures):
        os.makedirs(f_figures)

    sns.scatterplot(x=complexities, y=accs)
    plt.title("Rule set complexity / Validation accuracy ")
    plt.xlabel("Complexity")
    plt.ylabel("Accuracy")
    plt.xlim(left=-1, right=np.max(complexities) + np.min(complexities) + 1)
    plt.ylim(bottom=-0.05, top=1.05)
    plt.savefig(f_figures + f"compl_acc")

    steepnesses = [sample["steepness"] for sample in sampler]
    # sns.scatterplot(x=steepnesses, y=fidelities)
    ans = []
    unique_st = list(set(steepnesses))
    unique_st.sort()
    ans = {}
    for curr_steep in unique_st:
        ans[str(curr_steep)] = [
            a for idx, a in enumerate(fidelities) if steepnesses[idx] == curr_steep
        ]

    sns.boxplot(ans)
    plt.title("Tanh activation steepness / Fidelity NN - rule set")
    plt.xlabel("Steepness")
    plt.ylabel("Fidelity")
    plt.ylim(bottom=0.5, top=1.05)
    plt.savefig(f_figures + f"steepness_fid")

    l1s = [sample["l1"] for sample in sampler]
    sns.scatterplot(x=l1s, y=complexities)
    plt.title("L1 regularization factor / Rule set complexity")
    plt.xlabel("lambda1")
    plt.ylabel("Complexity")
    plt.xlim(left=np.min(l1s) - 0.001, right=np.max(l1s) + 0.001)
    plt.ylim(bottom=-1, top=np.max(complexities) + np.min(complexities) + 1)
    plt.savefig(f_figures + f"l1_compl")

    n_layers = [sample["n_layer"] for sample in sampler]
    unique_st = list(set(n_layers))
    unique_st.sort()
    ans = {}
    for curr_steep in unique_st:
        ans[str(curr_steep)] = [
            a for idx, a in enumerate(fidelities) if n_layers[idx] == curr_steep
        ]

    plt.figure()
    sns.boxplot(ans)
    plt.title("Number of fully connected Layers / Fidelity NN - rule set")
    plt.xlabel("No. layers")
    plt.ylabel("Fidelity")
    plt.ylim(bottom=0.5, top=1.05)
    plt.savefig(f_figures + f"nlayer_fid")

    print(f"Best found rule set: {best_model}")
    print(best_model.complexity())

    # with the new-found hyperparameter settings, we do one final cross validation
    re_clf.set_params(**best_params)
    re_cv_scores = cross_val_score(
        re_clf, X, y, cv=5, scoring="accuracy", error_score="raise"
    )
    print(f"cross validation scores: {re_cv_scores}")
    print(f"Mean validation acc: {re_cv_scores.mean()}")
    print(f"Standard deviation: {re_cv_scores.std()}")

    # Draw graphs
    # Best parameters:
    # abcdefg:
    # re_clf.set_params(l1=1e-04, layer_width=8, lr=0.003, n_layer=2, steepness=6)
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
