import copy
import os
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
import torch
from pandas import DataFrame, Series
import wittgenstein as lw
from sklearn import tree
from sklearn.metrics import accuracy_score, make_scorer

from sklearn.model_selection import (
    ParameterSampler,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
    KFold
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


def b_plot(x: list, y: list, x_label: str, y_label: str, title="", y_limits: tuple[float, float] | None = None):
    unique_x = list(set(x))
    unique_x.sort()
    ans = {}
    for x_val in unique_x:
        ans[str(x_val)] = [
            a for idx, a in enumerate(y) if x[idx] == x_val
        ]

    sns.boxplot(ans)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if y_limits is None:
        y_range = max(y) - min(y)
        y_limits = (min(y) - 0.05* y_range, max(y) + 0.05 * y_range)
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

def ripper_model_complexity(ripper_clf: lw.RIPPER) -> int:
    out_model = str([str(rule) for rule in ripper_clf.ruleset_.rules])
    n_ors = out_model.count(",")
    n_ands = out_model.count("^")
    return 1 + n_ors + n_ands

def re_clf_complexity(re_clf: RuleExtractionClassifier) -> int:
    return re_clf.bool_graph.complexity()


def cross_validate(X: DataFrame, y: Series, clf: RuleExtractionClassifier, compl_func: callable):
    accs, compls = [], []
    kf = KFold(n_splits = 10)
    for _, (train_ids, valid_ids) in enumerate(kf.split(X)): 
        Xtrain = X.loc[train_ids]
        ytrain = y.loc[train_ids]
        Xvalid = X.loc[valid_ids]
        yvalid = y.loc[valid_ids]

        clf.fit(Xtrain, ytrain)
        accs.append(accuracy_score(clf.predict(Xvalid), yvalid))
        compls.append(compl_func(clf))
    return np.array(accs), np.array(compls)


def find_best_params(X, y, re_clf: RuleExtractionClassifier, param_grid: dict[str, list], n_iter: int, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # do a random search over the hyperparameter space
    sampler = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=seed))

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
    return sampler, accs, complexities, fidelities


def print_metrics(accs, compls):
    print(f"\tAccs: {accs}")
    print(f"\tMean acc: {accs.mean()}")
    print(f"\tacc std: {accs.std()}")
    print(f"\tcomplexities: {compls}")
    print(f"\tMean acc: {compls.mean()}")
    print(f"\tacc std: {compls.std()}")

def stats(arr) -> str:
    return(f"{arr}\t|\tmean = {np.mean(arr)}\t|\tstd = {np.std(arr)}")

def training_runs(key):
    seed = 1
    set_seed(seed)

    X, y = load_dataframe(key)

    # find best hyperparameter settings
    param_grid = {
        "layer_width": [12],
        "lr": [3e-4, 1e-3, 4e-3],
        "l1": [0.0],
        "n_layer": [1, 2, 3],
        "steepness": [2, 4, 8],
    }

    re_clf = RuleExtractionClassifier(
        lr=4e-3,
        layer_width=10,
        n_layer=1,
        l1=4e-3,
        epochs=8000,
        wd=0.0,
        steepness=4,
        delay=800,
    )

    n_iter = 20
    print(f"Starting Hyperparameter search...")
    sampler, accs, complexities, fidelities = find_best_params(X, y, re_clf, param_grid, n_iter, seed)
    scores = custom_score(accs, complexities)
    best_idx = np.argmax(scores)
    best_params = sampler[best_idx]

    print(f"Hyperparameter search done.")

    print(f"Grid Search Accuracies: {accs = }")
    print(f"Grid Search Complexities: {complexities = }")
    print(f"Grid Search Scores:  {scores = }")
    print(f"Best hyperparameters: {best_params}\n")

    f_figures = f"figures/{key}/"
    if not os.path.isdir(f_figures):
        os.makedirs(f_figures)

    sns.scatterplot(x=complexities, y=accs)
    plt.title("Rule set complexity / Validation accuracy")
    plt.xlabel("Complexity")
    plt.ylabel("Accuracy")
    plt.xlim(left=-1, right=np.max(complexities) + np.min(complexities) + 1)
    plt.ylim(bottom=-0.05, top=1.05)
    plt.savefig(f_figures + f"compl_acc")

    steepnesses = [sample["steepness"] for sample in sampler]
    b_plot(steepnesses, fidelities, "Steepness", "Fidelity", title="Steepness / Fidelity NN - rule set")
    plt.savefig(f_figures + f"steepness_fid")
    plt.figure()

    l1s = [sample["l1"] for sample in sampler]
    b_plot(l1s, complexities, "L1 coefficient", "Complexity", title="L1 coefficient / rule set complexity")
    plt.savefig(f_figures + f"l1_compl")
    plt.figure()

    n_layers = [sample["n_layer"] for sample in sampler]
    b_plot(n_layers, fidelities, "Num of layers", "Fidelity", title="Number of neural net layers / Fidelity NN - rule set")
    plt.savefig(f_figures + f"nlayer_fid")
    plt.figure()
    
    ripper_clf = lw.RIPPER()
    ripper_accs, ripper_compls = cross_validate(X, y, ripper_clf, ripper_model_complexity)
    ripper_scores = custom_score(ripper_accs, ripper_compls)
    best_ripper_idx = np.argmax(ripper_scores)
    best_ripper_acc = ripper_accs[best_ripper_idx]
    best_ripper_compl = ripper_compls[best_ripper_idx]


    re_clf.set_params(**best_params)
    re_accs, re_compls = cross_validate(X, y, re_clf, re_clf_complexity)
    re_scores = custom_score(re_accs, re_compls)
    best_re_idx = np.argmax(re_scores)
    best_re_acc = re_accs[best_re_idx]
    best_re_compl = re_compls[best_re_idx]
    
    n = len(ripper_accs)
    data = np.zeros((2 * n, 3))
    data[:n, 0] = ripper_accs
    data[n:, 0] = re_accs
    data[:n, 1] = ripper_compls
    data[n:, 1] = re_compls
    results = pd.DataFrame(columns=["acc", "compl", "Classifier"])
    results['acc'] = np.concatenate([re_accs, ripper_accs])
    results['compl'] = np.concatenate([re_compls, ripper_compls])
    results['Classifier'] = ["DRE" for _ in range(n)] + ["RIPPER" for _ in range(n)]

    print("------------- RIPPER ---------------")
    print(f"Accuracy: {stats(ripper_accs)}")
    print(f"Complexity: {stats(ripper_compls)}")
    print(f"Best model acc: {best_ripper_acc}")
    print(f"Best model comply: {best_ripper_compl}")


    print("------------- RULE EXTRACTION ---------------")
    print(f"Accuracy: {stats(re_accs)}")
    print(f"Complexity: {stats(re_compls)}")
    print(f"Best model acc: {best_re_acc}")
    print(f"Best model comply: {best_re_compl}")
    

    sns.scatterplot(results, x="compl", y="acc", hue="Classifier")
    plt.title("Comparison DRE - RIPPER")
    plt.xlabel("Complexity")
    plt.ylabel("validation Accuracy")
    plt.savefig(f_figures + f"cross_val_comparison")
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


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Train an MLP on a binary classification task with an ADAM optimizer."
    )
    parser.add_argument("dataset")
    try:
        return parser.parse_args().dataset
    except:
        return "abcdefg"


if __name__ == "__main__":
    ds = get_arguments()
    main(ds)
