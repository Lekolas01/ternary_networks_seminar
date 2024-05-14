from argparse import ArgumentParser, Namespace
from collections.abc import MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from neuron import NeuronGraph
from q_neuron import QuantizedNeuronGraph, QuantizedNeuronGraph2
from rule_set import RuleSetGraph


def nn_to_rule_set(
    model: nn.Sequential, data: MutableMapping[str, np.ndarray], vars: Sequence[str]
):
    # transform the trained neural network to a directed graph of full-precision neurons
    neuron_graph = NeuronGraph.from_nn(model, vars)
    # transform the graph to a new graph of perceptrons with quantized step functions
    # q_neuron_graph = QuantizedNeuronGraph.from_neuron_graph(neuron_graph, data)
    q_neuron_graph = QuantizedNeuronGraph2.from_neuron_graph(neuron_graph, data)

    print(neuron_graph)
    print(q_neuron_graph)

    # transform the quantized graph to a set of if-then rules
    bool_graph = RuleSetGraph.from_q_neuron_graph(q_neuron_graph)
    return (neuron_graph, q_neuron_graph, bool_graph)


def main(model_path: str, data_path: str):
    print(f"Loading trained model from {model_path}...")
    # get the last updated model
    print(f"{model_path = }")
    model = torch.load(model_path)

    df = pd.read_csv(data_path, dtype=float)
    keys = list(df.columns)
    keys.pop()  # remove target column
    y = np.array(df["target"])
    ng_data = {key: np.array(df[key], dtype=float) for key in keys}
    nn_data = np.stack([ng_data[key] for key in keys], axis=1)
    nn_data = torch.Tensor(nn_data)
    bg_data = {key: np.array(df[key], dtype=bool) for key in keys}

    print("Transforming model to rule set...")
    ng, q_ng, bg = nn_to_rule_set(model, ng_data, keys)

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


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Given an already trained MLP and the dataset it was trained on, extract a set of if-then rules with similar behavior as the MLP."
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
    main(args.model_path, args.data_path)
