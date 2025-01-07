import copy
import os
import sys
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wittgenstein as lw
from pandas import DataFrame, Series
from sklearn import tree
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, ParameterSampler, train_test_split

from datasets import FileDataset, get_df
from rule_extraction import nn_to_rule_set
from rule_extraction_classifier import RuleExtractionClassifier
from utilities import accuracy, set_seed

np.set_printoptions(suppress=True)


def custom_score(acc, complexity):
    return (acc - 0.5) ** 2 / complexity


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


def b_plot(
    x: list,
    y: list,
    x_label: str,
    y_label: str,
    title="",
    y_limits: tuple[float, float] | None = None,
):
    unique_x = list(set(x))
    unique_x.sort()
    ans = {}
    for x_val in unique_x:
        ans[str(x_val)] = [a for idx, a in enumerate(y) if x[idx] == x_val]

    sns.boxplot(ans)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_limits is None:
        y_range = max(y) - min(y)
        y_limits = (min(y) - 0.05 * y_range, max(y) + 0.05 * y_range)
    plt.ylim(bottom=y_limits[0], top=y_limits[1])


def load_dataframe(key: str) -> tuple[DataFrame, Series]:
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


def rip_compl(ripper_clf: lw.RIPPER) -> int:
    out_model = str([str(rule) for rule in ripper_clf.ruleset_.rules])
    n_ors = out_model.count(",")
    n_ands = out_model.count("^")
    return 1 + n_ors + n_ands


def rip_model_str(ripper_clf: lw.RIPPER):
    ans = str([str(rule) for rule in ripper_clf.ruleset_.rules]).replace(", ", "\n")
    return ans


def re_clf_complexity(re_clf: RuleExtractionClassifier) -> int:
    return re_clf.bool_graph.complexity() + 1


def cross_val_and_log(X, y, clf, compl_f, model_str_f, n_splits, f_results, alg_name):
    accs, compls, models = cross_validate(X, y, clf, compl_f, model_str_f, n_splits)
    scores = custom_score(accs, compls)
    best_idx = np.argmax(scores)
    best_acc = accs[best_idx]
    best_compls = compls[best_idx]
    best_model = models[best_idx]

    with open(f_results, "a") as text_file:
        text_file.write(f"------------- {alg_name} ---------------\n")
        text_file.write(f"\tAccuracy: {stats(accs)}\n")
        text_file.write(f"\tComplexity: {stats(compls)}\n")
        text_file.write(f"\tBest model acc: {best_acc}\n")
        text_file.write(f"\tBest model comply: {best_compls}\n")
        text_file.write(f"\tBest model: {best_model}\n\n")
    return accs, compls


def cross_validate(
    X: DataFrame, y: Series, clf: BaseEstimator, compl_func, model_str_func, n_splits=10
):
    accs, compls, models = [], [], []
    kf = KFold(n_splits=n_splits)
    for _, (train_ids, valid_ids) in enumerate(kf.split(X)):
        Xtrain = X.loc[train_ids]
        ytrain = y.loc[train_ids]
        Xvalid = X.loc[valid_ids]
        yvalid = y.loc[valid_ids]

        clf.fit(Xtrain, ytrain)
        accs.append(accuracy_score(clf.predict(Xvalid), yvalid))
        compls.append(compl_func(clf))
        models.append(model_str_func(clf))
    return np.array(accs), np.array(compls), models


def find_best_params(
    X,
    y,
    re_clf: RuleExtractionClassifier,
    param_grid: dict[str, list],
    n_iter: int,
    seed=0,
):
    n_folds = 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # do a random search over the hyperparameter space
    sampler = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=seed))

    accs, complexities, fidelities = (
        np.zeros((n_iter, n_folds)),
        np.zeros((n_iter, n_folds)),
        np.zeros((n_iter, n_folds)),
    )
    for i, sample in enumerate(sampler):
        re_clf.set_params(**sample)
        for fold in range(n_folds):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed + fold
            )
            re_clf.fit(X_train, y_train)
            accs[i, fold] = accuracy_score(y_test, re_clf.predict(X_test))
            complexities[i, fold] = re_clf_complexity(re_clf)
            fidelities[i, fold] = re_clf.fid_rule_set

    a = accs.mean(axis=1)
    c = complexities.mean(axis=1)
    f = fidelities.mean(axis=1)
    s = custom_score(accs, complexities).mean(axis=1)
    return (sampler, a, c, f, s)


def print_metrics(accs, compls):
    print(f"\tAccs: {accs}")
    print(f"\tMean acc: {accs.mean()}")
    print(f"\tacc std: {accs.std()}")
    print(f"\tcomplexities: {compls}")
    print(f"\tMean acc: {compls.mean()}")
    print(f"\tacc std: {compls.std()}")


def stats(arr) -> str:
    return f"\t{arr}\n\tmean = {np.mean(arr)}\n\tstd = {np.std(arr)}\n"


def training_runs(key):
    seed = 1
    set_seed(seed)

    X, y = load_dataframe(key)
    f_outputs = f"output/{key}/"
    f_results = f_outputs + f"results.txt"

    if not os.path.isdir(f_outputs):
        os.makedirs(f_outputs)

    # find best hyperparameter settings
    param_grid = {
        "layer_width": [10, 6],
        "lr": [3e-4, 1e-3, 3e-3],
        "l1": [1e-5, 4e-4, 3e-3],
        "n_layer": [1, 2],
        "steepness": [4, 8, 16],
    }

    delay = 600

    re_clf = RuleExtractionClassifier(
        lr=4e-3,
        layer_width=10,
        n_layer=1,
        l1=4e-3,
        epochs=8000,
        wd=0.0,
        steepness=4,
        delay=delay,
        verbose=True,
    )

    n_iter = 5
    print(f"Starting Hyperparameter search...")
    sampler, accs, complexities, fidelities, scores = find_best_params(
        X, y, re_clf, param_grid, n_iter, seed
    )
    best_idx = np.argmax(scores)
    best_params = sampler[best_idx]

    print(f"Hyperparameter search done.")

    print(f"Grid Search Accuracies: {accs = }")
    print(f"Grid Search Complexities: {complexities = }")
    print(f"Grid Search Scores:  {scores = }")
    print(f"Best hyperparameters: {best_params}\n")

    sns.scatterplot(x=complexities, y=accs)
    plt.title("Rule set complexity / Validation accuracy")
    plt.xlabel("Complexity")
    plt.ylabel("Accuracy")
    plt.xlim(left=-1, right=np.max(complexities) + np.min(complexities) + 1)
    plt.ylim(bottom=-0.05, top=1.05)
    plt.savefig(f_outputs + f"compl_acc")

    steepnesses = [sample["steepness"] for sample in sampler]
    b_plot(
        steepnesses,
        fidelities,
        "Steepness",
        "Fidelity",
        title="Steepness / Fidelity NN - rule set",
    )
    plt.savefig(f_outputs + f"steepness_fid")
    plt.figure()

    l1s = [sample["l1"] for sample in sampler]
    b_plot(
        l1s,
        complexities,
        "L1 coefficient",
        "Complexity",
        title="L1 coefficient / rule set complexity",
    )
    plt.savefig(f_outputs + f"l1_compl")
    plt.figure()

    n_layers = [sample["n_layer"] for sample in sampler]
    b_plot(
        n_layers,
        fidelities,
        "Num of layers",
        "Fidelity",
        title="Number of neural net layers / Fidelity NN - rule set",
    )
    plt.savefig(f_outputs + f"nlayer_fid")
    plt.figure()

    with open(f_results, "w") as of:

        of.write("--- Hyperparameter search ---\n")
        of.write(f"\t{param_grid = }\n\n")
        of.write(f"\t{accs = }\n")
        of.write(f"\t{complexities = }\n")
        of.write(f"\t{scores = }\n")
        of.write(f"\t{best_idx = }\n")
        of.write(f"\t{best_params = }\n\n")

    k = 10
    # delay = 1000

    ripper_clf = lw.RIPPER()
    ripper_accs, ripper_compls = cross_val_and_log(
        X, y, ripper_clf, rip_compl, rip_model_str, k, f_results, "RIPPER"
    )

    re_clf.set_params(**best_params, delay=delay)
    re_accs, re_compls = cross_val_and_log(
        X, y, re_clf, re_clf_complexity, lambda clf: clf.bool_graph, k, f_results, "DRE"
    )

    data = np.zeros((2 * k, 3))
    data[:k, 0] = ripper_accs
    data[k:, 0] = re_accs
    data[:k, 1] = ripper_compls
    data[k:, 1] = re_compls
    results = pd.DataFrame(columns=["acc", "compl", "Classifier"])
    results["acc"] = np.concatenate([re_accs, ripper_accs])
    results["compl"] = np.concatenate([re_compls, ripper_compls])
    results["Classifier"] = ["DRE" for _ in range(k)] + ["RIPPER" for _ in range(k)]

    sns.scatterplot(results, x="compl", y="acc", hue="Classifier")
    plt.title("Comparison DRE - RIPPER")
    plt.xlabel("Complexity")
    plt.ylabel("validation Accuracy")
    plt.savefig(f_outputs + f"cross_val_comparison")
    plt.figure()

    exit()


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


def get_arguments() -> str:
    parser = ArgumentParser(
        description="Train an MLP on a binary classification task with an ADAM optimizer."
    )
    parser.add_argument("dataset")
    return parser.parse_args().dataset


if __name__ == "__main__":
    ds = get_arguments()
    main(ds)
