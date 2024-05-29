import copy
import itertools
import os
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from ckmeans_1d_dp import ckmeans
from torch.utils.data.dataloader import DataLoader

from bool_formula import Activation
from datasets import FileDataset
from generate_parity_dataset import parity_df
from models.model_collection import ModelFactory, NNSpec
from rule_extraction import nn_to_rule_set
from rule_set import QuantizedLayer
from train_mlp import train_mlp
from train_model import validate
from utilities import accuracy, set_seed

# Grid Search over NN training hyperparameters
#   call train_mlp.py
# For each trained model, extract rule set
# save all rule sets
# create graph for rule sets: dimensions are complexity and accuracy


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


def main(k: int):
    seed = 1
    set_seed(seed)
    # Generate dataframe for parity
    f_root = f"runs/parity{k}"
    f_data = f"{f_root}/data.csv"
    f_models = f"{f_root}/models"
    f_runs = f"{f_root}/runs.csv"
    f_losses = f"{f_root}/losses.csv"

    os.makedirs(f_models, exist_ok=True)
    df = parity_df(k=k, shuffle=False, n=1024)
    # write dataset to data.csv
    df.to_csv(f_data, index=False)
    df = pd.read_csv(f_data)
    print(f"Generated dataset with shape {df.shape}")

    # Do a single NN training run on this dataset
    max_epochs = 6000
    bs = 64
    wd = 0.0

    lrs = [1e-3, 3e-3, 1e-2, 3e-2]
    l1s = [1e-5, 1e-4]
    n_layers = [2, 3, 4]
    runs = pd.DataFrame(
        columns=["idx", "lr", "n_layer", "seed", "epochs", "bs", "l1", "wd"]
    )
    n_runs = len(l1s) * len(lrs) * len(n_layers)
    for idx, (l1, lr, n_layer) in enumerate(itertools.product(l1s, lrs, n_layers)):
        print(f"{max_epochs = }\t|{bs = }\t|{wd = }\t|{lr = }\t|{n_layer = }\t|{l1 = }")
        model_path = f"{f_models}/{idx}.pth"
        # if idx == 0:
        #     continue
        # full_dl = DataLoader(FileDataset(df), batch_size=df.shape[0], shuffle=True)
        # model = torch.load(f"{f_models}/{idx}.pth")
        # keys = list(df.columns)
        # keys.pop()
        # X, y = next(iter(full_dl))
        # y = y.detach().numpy()
        # ng_data = {key: np.array(df[key], dtype=float) for key in keys}
        # ng, q_ng, bg = nn_to_rule_set(model, ng_data, keys)
        # print(f"{accuracy(model, full_dl, 'cpu')}")
        # model = quantize_first_lin_layer(model, df)
        # print(model[0])
        # print(model[0](X))
        # print(f"{accuracy(model, full_dl, 'cpu')}")
        # ng_data = {key: np.array(df[key], dtype=float) for key in keys}
        # q_ng_pred = q_ng(ng_data)
        # # print(f"{q_ng_pred = }")
        # print(f"{ng_data['h1'] = }")
        # q_ng_acc = 1 - sum(abs(q_ng_pred - y)) / len(q_ng_pred)
        # print(f"q_ng_acc = {q_ng_acc}")

        if os.path.isfile(model_path):
            model = torch.load(model_path)
            print("Model already trained. Skipping this training session...")
            continue
        spec: NNSpec = [(k - i, k - i - 1, Activation.TANH) for i in range(n_layer)]
        spec.append(((k - n_layer, 1, Activation.SIGMOID)))

        model = ModelFactory.get_model_by_spec(spec)

        all_metrics = []
        for i in range(n_layer):  # for each layer from left to right:
            # train a neural net
            metrics, dl, full_dl = train_mlp(
                df, model, seed, bs, lr, max_epochs, l1, wd
            )
            if i == 0:
                torch.save(model, f"{f_models}/full_precision_{idx}.pth")
                torch.save(model, model_path)
                keys = list(df.columns)
                keys.pop()
                y = np.array(df["target"])
                ng_data = {key: np.array(df[key], dtype=float) for key in keys}
                ng, q_ng, bg = nn_to_rule_set(model, ng_data, keys)
                ng_data = {key: np.array(df[key], dtype=float) for key in keys}
                q_ng_pred = q_ng(ng_data)
                q_ng_acc = 1 - sum(abs(q_ng_pred - y)) / len(q_ng_pred)
                print(f"old q_ng_acc = {q_ng_acc}")
                other_model = copy.deepcopy(model)
                while any(isinstance(l, nn.Linear) for l in other_model):
                    other_model = quantize_first_lin_layer(other_model, df)
                print(
                    f"immediately quantized model acc: {accuracy(other_model, full_dl, 'cpu')}"
                )
            print(f"accuracy before layer quant. = {accuracy(model, full_dl, 'cpu')}")
            model = quantize_first_lin_layer(model, df)
            print(f"accuracy after layer quant. = {accuracy(model, full_dl, 'cpu')}")
            all_metrics.append(metrics)

        # quantize the last layer
        model = quantize_first_lin_layer(model, df)

        # merge metrics together
        metrics = pd.concat(all_metrics)

        # append losses to losses.csv
        metrics.insert(loc=0, column="idx", value=idx)
        metrics.insert(
            loc=1,
            column="epoch",
            value=[i + 1 for i in range(len(metrics["train_loss"]))],
        )
        metrics.to_csv(f_losses, mode="a", index=False, header=(idx == 0))
        cols = [idx, lr, n_layer, seed, max_epochs, bs, l1, wd]
        runs.loc[idx] = [str(val) for val in cols]

        # append run config to run file
        runs.to_csv(f_runs, mode="a", header=(idx == 0), index=False)

        # save quantized model
        torch.save(model, model_path)
        print(f"accuracy after final quant. = {accuracy(model, full_dl, 'cpu')}")
        print()

    complexities = []
    accs = []
    df_runs = pd.read_csv(f_runs)
    for idx in range(n_runs):
        model_path = f"{f_models}/{idx}.pth"
        model = torch.load(model_path)
        keys = list(df.columns)
        keys.pop()
        y = np.array(df["target"])
        ng_data = {key: np.array(df[key], dtype=float) for key in keys}
        _, q_ng, bg = nn_to_rule_set(model, ng_data, keys)
        bg_data = {key: np.array(data, dtype=bool) for key, data in ng_data.items()}
        q_ng_pred = bg(bg_data)
        complexities.append(bg.complexity())
        acc = 1 - sum(abs(q_ng_pred - y)) / len(q_ng_pred)
        accs.append(acc)
        print(
            f"compl = {bg.complexity()}\tl1 = {df_runs['l1'].iloc[idx]}\tacc = {acc}\tlr = {df_runs['lr'].iloc[idx]}"
        )

    # TODO für morgen:
    #   Regeln lernen und miteinander vergleichen können
    sns.scatterplot(x=complexities, y=accs)
    plt.show()

    df_metrics = pd.read_csv(f_losses)
    for i in range(n_runs):
        run_info = df_runs.iloc[i]
        temp = df_metrics[df_metrics["idx"] == i]
        title = f"l1: {run_info['l1']}   |   layers: {run_info['n_layer']}   |   lr: {run_info['lr']}"
        sns.lineplot(x=[i + 1 for i in range(temp.shape[0])], y=temp["train_loss"])
        plt.title(title)
        plt.show()

    # find "best" models in terms of complexity and validation accuracy


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Train an MLP on a binary classification task with an ADAM optimizer."
    )
    parser.add_argument("k", help="Arity")
    parser.add_argument(
        "--root",
        help="Root folder for all the run info.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(int(args.k))
