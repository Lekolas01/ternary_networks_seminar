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
from C45 import C45Classifier
from ckmeans_1d_dp import ckmeans
from genericpath import isfile
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data.dataloader import DataLoader

from bool_formula import Activation, overlap
from datasets import FileDataset, get_df
from generate_parity_dataset import parity_df
from models.model_collection import ModelFactory, NNSpec
from q_neuron import QuantizedLayer
from rule_extraction import nn_to_rule_set
from train_mlp import train_mlp
from utilities import accuracy, set_seed

# Grid Search over NN training hyperparameters
#   call train_mlp.py
# For each trained model, extract rule set
# save all rule sets
# create graph for rule sets: dimensions are complexity and accuracy


class RuleExtractionClassifier:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass


def quantize_first_lin_layer(model: nn.Sequential, df: pd.DataFrame) -> nn.Sequential:
    full_dl = DataLoader(FileDataset(df), batch_size=df.shape[0], shuffle=True)
    lin_layer_indices = [
        i for i in range(len(model)) if isinstance(model[i], nn.Linear)
    ]
    lin_layer_idx = lin_layer_indices[0]
    assert lin_layer_idx >= 0

    q_layer = quantize_layer(model, lin_layer_idx, len(lin_layer_indices) == 1, full_dl)
    model = nn.Sequential(
        *[model[i] for i in range(lin_layer_idx)],
        q_layer,
        *[model[i] for i in range(2 + lin_layer_idx, len(model))],
    )
    return model


def quantize_layer(
    model: nn.Sequential, lin_layer_idx: int, is_last: bool, dl: DataLoader
) -> QuantizedLayer:
    lin_layer: nn.Linear = model[lin_layer_idx]  # type: ignore
    assert isinstance(lin_layer, nn.Linear)
    act = model[lin_layer_idx + 1]
    lin_layer.requires_grad_(False)

    if is_last:
        assert isinstance(act, nn.Sigmoid)
        return QuantizedLayer(lin_layer, torch.tensor(0.0), torch.tensor(1.0))

    X, y = next(iter(dl))
    for i in range(lin_layer_idx):
        X = model[i](X)

    y_hat: torch.Tensor = act(lin_layer(X)).detach().numpy()
    x_thrs = torch.Tensor(lin_layer.out_features)
    y_low = torch.Tensor(lin_layer.out_features)
    y_high = torch.Tensor(lin_layer.out_features)
    assert isinstance(act, nn.Tanh)
    for i in range(lin_layer.out_features):
        ck_ans = ckmeans(y_hat[:, i], (2))
        cluster = ck_ans.cluster
        max_0 = np.max(y_hat[:, i][cluster == 0])
        min_1 = np.min(y_hat[:, i][cluster == 1])
        y_thr = (max_0 + min_1) / 2
        x_thr = np.arctanh(y_thr)
        x_thrs[i] = x_thr
        y_low[i] = ck_ans.centers[0]
        y_high[i] = ck_ans.centers[1]
    with torch.no_grad():
        lin_layer.bias -= x_thrs  # type: ignore
    return QuantizedLayer(lin_layer, y_low, y_high)


def first_linear_layer(model: nn.Sequential) -> int:
    for idx, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            return idx
    return -1


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

    # Do a single NN training run on this dataset
    max_epochs = 5000
    bs = 64
    wd = 0.0

    ks = [5]
    lrs = [1e-2, 1e-3]
    l1s = [0.0, 1e-5, 3e-3]
    n_layers = [1, 2]
    runs = pd.DataFrame(
        columns=["idx", "seed", "lr", "k", "n_layer", "l1", "epochs", "wd"]
        + ["train_loss", "nn_acc", "bg_acc", "fidelity", "complexity"]
    )
    all_metrics = []
    n_runs = len(l1s) * len(lrs) * len(n_layers)
    bgs = []

    for idx, (lr, k, n_layer, l1) in enumerate(
        itertools.product(lrs, ks, n_layers, l1s)
    ):
        print(
            f"{idx = }\t{max_epochs = }\t|{bs = }\t|{wd = }\t|{k = }\t{l1 = }\t|{lr = }\t|{n_layer = }"
        )
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

    if retrain:
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

    df_metrics = pd.read_csv(f_losses)
    for i in range(n_runs):
        run_info = runs.iloc[i]
        temp = df_metrics[df_metrics["idx"] == i]
        title = f"l1: {run_info['l1']}   |   layers: {run_info['n_layer']}   |   lr: {run_info['lr']}"
        sns.lineplot(x=[i + 1 for i in range(temp.shape[0])], y=temp["train_loss"])
        plt.title(title)
        # plt.show()

    print(f"")
    print("------------------ CART classifier ------------------")

    # Train classifier
    dtree = DecisionTreeClassifier()
    X = df.drop(["target"], axis=1)
    y = df["target"]
    dtree = dtree.fit(X, y)
    _ = plt.figure(figsize=(10, 10), dpi=240)
    tree.plot_tree(dtree, feature_names=list(df.columns))
    dtree.predict(X)
    plt.show()


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Train an MLP on a binary classification task with an ADAM optimizer."
    )
    parser.add_argument("dataset")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args.dataset)
