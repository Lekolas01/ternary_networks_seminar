import os
from argparse import ArgumentParser, Namespace
from collections.abc import MutableMapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from neuron import NeuronGraph
from q_neuron import QNG_from_QNN, QuantizedLayer, QuantizedNeuronGraph2
from rule_set import RuleSetGraph


def nn_to_rule_set(model: nn.Sequential, df: pd.DataFrame):
    keys = list(
        df.columns[:-1]
    )  # assume that df only has one target column at the far right
    data = {key: np.array(df[key], dtype=float) for key in keys}
    q_neuron_graph: QuantizedNeuronGraph2
    if isinstance(model[0], nn.Linear):
        # transform the trained neural network to a directed graph of full-precision neurons
        neuron_graph = NeuronGraph.from_nn(model, keys)
        # transform the graph to a new graph of perceptrons with quantized step functions
        # q_neuron_graph = QuantizedNeuronGraph.from_neuron_graph(neuron_graph, data)
        q_neuron_graph = QuantizedNeuronGraph2.from_neuron_graph(neuron_graph, data)
    else:
        assert isinstance(model[0], QuantizedLayer)
        q_neuron_graph = QNG_from_QNN(model, keys)

    # transform the quantized graph to a set of if-then rules
    bool_graph = RuleSetGraph.from_q_neuron_graph(q_neuron_graph)
    return (q_neuron_graph, bool_graph, data)


def inspect_model(f_model: str, f_data: str):
    print(f"Loading trained model from {f_model}...")
    # get the last updated model
    model = torch.load(f_model)

    print(f"Loading dataset from {f_data}...")
    df = pd.read_csv(f_data, dtype=float)
    keys = list(df.columns)
    keys.pop()  # remove target column
    y = np.array(df["target"])
    ng_data = {key: np.array(df[key], dtype=float) for key in keys}
    nn_data = np.stack([ng_data[key] for key in keys], axis=1)
    nn_data = torch.Tensor(nn_data)
    bg_data = {key: np.array(df[key], dtype=bool) for key in keys}

    print("Transforming model to rule set...")
    q_ng, bg = nn_to_rule_set(model, ng_data, keys)

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    head_len = 4

    print("----------------- MLP -----------------")
    print(f"{model = }")
    nn_out = model(nn_data).detach().numpy()
    nn_pred = np.round(nn_out)
    print("mean absolute error: ", np.mean(np.abs(nn_out - y)))
    print("prediction accuracy:\t", np.array(1.0) - np.mean(np.abs(nn_pred - y)))

    print("----------------- Neuron Graph -----------------")
    print(f"{ng = }")
    ng_out = ng(ng_data)
    ng_pred = np.round(ng_out)

    print("mean absolute error: ", np.mean(np.abs(ng_out - y)))
    print("prediction accuracy:\t", np.array(1.0) - np.mean(np.abs(ng_pred - y)))

    print("------------- Quantized NG -------------")

    print(f"{q_ng = }")
    q_ng_out = q_ng(ng_data)
    q_ng_pred = np.round(q_ng_out)

    print("mean absolute error: ", np.mean(np.abs(q_ng_out - y)))
    print("prediction accuracy:\t", np.array(1.0) - np.mean(np.abs(q_ng_pred - y)))

    print("----------------- Rule Set Graph -----------------")

    print(f"{bg = }")
    print(f"{bg.complexity() = }")
    bg_pred = bg(bg_data)
    bg_out = np.where(bg_pred == True, 1.0, 0.0)

    print(f"outs: {bg_out[:head_len] = }")
    # print(f"outs: {bg_out = }")
    print(f"preds: {bg_pred[:head_len] = }")
    # print(f"preds: {bg_pred = }")
    print("mean absolute error: ", np.mean(np.abs(bg_out - y)))
    print("prediction accuracy:\t", np.array(1.0) - np.mean(np.abs(bg_out - y)))

    print("----------------- Mean Errors -----------------")

    print("mean error nn - ng: ", np.mean(np.abs(nn_out - ng_out)))
    print("mean error ng - q_ng: ", np.mean(np.abs(ng_out - q_ng_out)))
    print(
        "mean error q_ng - norm_: ",
        np.mean(np.abs(nn_out - bg_out)),
    )

    print("----------------- Fidelity -----------------")

    print("fidelity nn - ng:\t", np.mean(nn_out.round() == ng_out.round()))
    print("fidelity ng - q_ng:\t", np.mean(ng_out.round() == q_ng_out.round()))
    print("fidelity nn - q_ng:\t", np.mean(nn_out.round() == q_ng_out.round()))
    print("fidelity nn - b_ng:\t", np.mean(nn_out.round() == bg_out))

    print("----------------- Final Acc. -----------------")

    print(
        "accuracy neuron graph:\t",
        np.array(1.0) - np.mean(np.abs(np.round(ng_out) - y)),
    )
    print(
        "accuracy quantized neuron graph:\t",
        np.array(1.0) - np.mean(np.abs(q_ng_out - y)),
    )
    print("accuracy boolean graph:\t", np.array(1.0) - np.mean(np.abs(bg_out - y)))

    print("All Errors:")
    models = [model, ng, q_ng, bg]
    outs = [nn_out, ng_out, q_ng_out, bg_out]
    preds = [nn_pred, ng_pred, q_ng_pred, bg_pred]
    n = len(outs)
    errors = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            errors[i, j] = np.mean(np.abs(outs[i] - outs[j]))

    print("Errors:")
    print("nn\tng\tq_ng\tn_ng\tbg")
    print(errors)

    fidelities = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            fidelities[i, j] = np.mean(preds[i] == preds[j])
    print("Fidelities:")
    print("nn\tng\tq_ng\tn_ng\tbg")
    print(fidelities)
    # Fidelity: the percentage of test examples for which the classification made by the rules agrees with the neural network counterpart
    # Accuracy: the percentage of test examples that are correctly classified by the rules
    # Consistency: is given if the rules extracted under different training sessions produce the same classifications of test examples
    # Comprehensibility: is determined by measuring the number of rules and the number of antecedents per rule
    pass


def rule_extraction_grid(f_models: str, f_data: str):
    df = pd.read_csv(f_data, dtype=float)
    keys = list(df.columns)
    keys.pop()  # remove target column
    y = np.array(df["target"])
    ng_data = {key: np.array(df[key], dtype=float) for key in keys}
    nn_data = np.stack([ng_data[key] for key in keys], axis=1)
    nn_data = torch.Tensor(nn_data)
    bg_data = {key: np.array(df[key], dtype=bool) for key in keys}

    files = [Path(f_models, f) for f in os.listdir(f_models) if f.endswith(".pth")]
    accs = np.empty((len(files), 4))
    bgs = []
    for idx, f_model in enumerate(files):
        model = torch.load(f_model)
        q_ng, bg = nn_to_rule_set(model, ng_data, keys)

        nn_out = model(nn_data).detach().numpy()
        nn_pred = np.round(nn_out)
        nn_acc = float(1.0 - np.mean(np.abs(nn_pred - y)))

        q_ng_out = q_ng(ng_data)
        q_ng_pred = np.round(q_ng_out)
        q_ng_acc = float(1.0 - np.mean(np.abs(q_ng_pred - y)))

        bg_out = bg(bg_data)
        bg_pred = np.where(bg_out == True, 1.0, 0.0)
        bg_acc = float(1.0 - np.mean(np.abs(bg_pred - y)))
        accs[idx] = np.array([nn_acc, q_ng_acc, bg_acc, bg.complexity()])
        bgs.append(bg)

    acc_df = pd.DataFrame(
        accs, columns=["nn_acc", "q_ng_acc", "bg_acc", "bg_complexity"]
    )
    return acc_df, bgs


def inspect_many_models(f_models: str, f_data: str):
    acc_df, _ = rule_extraction_grid(f_models, f_data)

    fig = plt.figure(figsize=(6, 6))
    # ax = Axes3D(fig)  # Method 1
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        acc_df["nn_acc"],
        acc_df["q_ng_acc"],
        acc_df["bg_acc"],
        c=acc_df["nn_acc"],
        marker="o",
    )
    ax.set_xlabel("NN Accuracy")
    ax.set_ylabel("Quantized NN Accuracy")
    ax.set_zlabel("Rule Set Accuracy")  # type: ignore
    plt.show()


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Given multiple trained MLPs and the dataset they were trained on, extract a set of if-then rules with similar behavior as the MLP."
    )
    parser.add_argument(
        "data_path",
        help="Relative path to the dataset file, starting from root folder.",
    )
    parser.add_argument(
        "model_path",
        help="Relative path to the .pth file, starting from root folder.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()

    acc_df, bgs = rule_extraction_grid(args.model_path, args.data_path)

    # ---------- plot 3D accuracies ----------
    fig = plt.figure(figsize=(6, 6))
    # ax = Axes3D(fig)  # Method 1
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        acc_df["nn_acc"],
        acc_df["q_ng_acc"],
        acc_df["bg_acc"],
        c=acc_df["nn_acc"],
        marker="o",
    )
    ax.set_xlabel("NN Accuracy")
    ax.set_ylabel("Quantized NN Accuracy")
    ax.set_zlabel("Rule Set Accuracy")  # type: ignore
    plt.show()

    # ---------- plot bg models with complexity ----------
    sns.scatterplot(acc_df, x="bg_complexity", y="bg_acc")
    plt.show()

    min_idx = np.argmin(acc_df["bg_complexity"])
    print(min_idx)
    print(bgs[min_idx])
    print(bgs[min_idx].complexity())
