import os
from argparse import ArgumentParser, Namespace
from collections.abc import MutableMapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bool_formula import PARITY
from datasets import FileDataset
from gen_data import gen_data
from models.model_collection import ModelFactory
from my_logging.loggers import LogMetrics, Tracker
from neuron import NeuronGraph
from q_neuron import QuantizedNeuronGraph, QuantizedNeuronGraph2
from rule_set import RuleSetGraph
from train_model import training_loop
from utilities import set_seed


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Train a neural network on a categorical dataset and then convert it to a rule set."
    )
    parser.add_argument(
        "--data",
        help="The name of the dataset (it must exist in the data/generated folder).",
    )
    parser.add_argument(
        "--model",
        help="The name of the neural net configuration - see models.model_collection.py.",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="If specified, the model will be retrained, even if it is already saved.",
    )
    return parser.parse_args()


def nn_to_rule_set(
    model: nn.Sequential,
    data: MutableMapping[str, np.ndarray],
    vars: Sequence[str],
    verbose=False,
):
    # transform the trained neural network to a directed graph of full-precision neurons
    neuron_graph = NeuronGraph.from_nn(model, vars)
    # transform the graph to a new graph of perceptrons with quantized step functions
    q_neuron_graph = QuantizedNeuronGraph.from_neuron_graph(
        neuron_graph, data, verbose=verbose
    )
    norm_q_neuron_graph = QuantizedNeuronGraph2.from_neuron_graph(neuron_graph, data)

    # transform the quantized graph to a set of if-then rules
    bool_graph = RuleSetGraph.from_q_neuron_graph(q_neuron_graph)
    return (neuron_graph, q_neuron_graph, norm_q_neuron_graph, bool_graph)


def main():
    args = get_arguments()
    seed = 1
    epochs = 3000
    batch_size = 64
    lr = 0.006
    weight_decay = 0.0
    l1 = 2e-6
    verbose = False
    data_name = args.data
    model_name = args.model if hasattr(args, "model") else data_name

    data_path = Path("data/generated") / f"{data_name}.csv"
    problem_path = Path(f"runs/{data_name}")
    model_path = problem_path / f"{model_name}.pth"

    if not os.path.isdir(problem_path):
        print(f"Creating new directory at {problem_path}...")
        os.mkdir(problem_path)

    if args.new or not os.path.isfile(model_path):
        seed = set_seed(seed)
        print(f"{seed = }")
        print(f"No pre-trained model found. Starting training...")
        model = ModelFactory.get_model(model_name)

        train_dl = DataLoader(
            FileDataset(data_path), batch_size=batch_size, shuffle=True
        )
        valid_dl = DataLoader(
            FileDataset(data_path), batch_size=batch_size, shuffle=True
        )
        loss_fn = nn.BCELoss()
        optim = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay,
        )
        tracker = Tracker()
        tracker.add_logger(
            LogMetrics(["timestamp", "epoch", "train_loss", "train_acc"], log_every=50)
        )

        losses = training_loop(
            model,
            loss_fn,
            optim,
            train_dl,
            valid_dl,
            epochs=epochs,
            lambda1=l1,
            tracker=tracker,
            device="cpu",
        )

        try:
            torch.save(model, model_path)
            print(f"Successfully saved model to {model_path}")
        except Exception as inst:
            print(f"Could not save model to {model_path}: {inst}")
    print(f"Loading trained model from {model_path}...")
    # get the last updated model
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
    ng, q_ng, norm_q_ng, bg = nn_to_rule_set(model, ng_data, keys, verbose=verbose)

    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)

    head_len = 4
    print(f"Target vector y: {y[:head_len]}")

    print("----------------- MLP -----------------")
    print(f"{model = }")
    nn_out = model(nn_data).detach().numpy()
    nn_pred = np.round(nn_out)
    print(f"outs: {nn_out[:head_len]}")
    print(f"preds: {nn_pred[:head_len]}")
    print("mean absolute error: ", np.mean(np.abs(nn_out - y)))
    print("prediction accuracy:\t", np.array(1.0) - np.mean(np.abs(nn_pred - y)))

    print("----------------- Neuron Graph -----------------")
    print(f"{ng = }")
    ng_out = ng(ng_data)
    ng_pred = np.round(ng_out)

    print(f"outs: {ng_out[:head_len]}")
    print(f"preds: {ng_pred[:head_len]}")
    print("mean absolute error: ", np.mean(np.abs(ng_out - y)))
    print("prediction accuracy:\t", np.array(1.0) - np.mean(np.abs(ng_pred - y)))

    print("------------- Quantized NG -------------")

    print(f"{q_ng = }")
    q_ng_out = q_ng(ng_data)
    q_ng_pred = np.round(q_ng_out)

    print(f"outs: {q_ng_out[:head_len]}")
    print(f"preds: {q_ng_pred[:head_len]}")
    print("mean absolute error: ", np.mean(np.abs(q_ng_out - y)))
    print("prediction accuracy:\t", np.array(1.0) - np.mean(np.abs(q_ng_pred - y)))

    print("------------- Normalized QNG -------------")

    print(f"{norm_q_ng = }")
    norm_q_ng_out = norm_q_ng(ng_data)
    norm_q_ng_pred = np.round(norm_q_ng_out)

    print(f"outs: {norm_q_ng_out[:head_len] = }")
    print(f"preds: {norm_q_ng_pred[:head_len] = }")
    print("mean absolute error: ", np.mean(np.abs(norm_q_ng_out - y)))
    print("prediction accuracy:\t", np.array(1.0) - np.mean(np.abs(norm_q_ng_pred - y)))

    print("----------------- Rule Set Graph -----------------")

    print(f"{bg = }")
    bg_pred = bg(bg_data)
    bg_out = np.where(bg_pred == True, 1.0, 0.0)

    # print(f"outs: {bg_out[:head_len] = }")
    print(f"outs: {bg_out = }")
    # print(f"preds: {bg_pred[:head_len] = }")
    print(f"preds: {bg_pred = }")
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

    # Fidelity: the percentage of test examples for which the classification made by the rules agrees with the neural network counterpart
    # Accuracy: the percentage of test examples that are correctly classified by the rules
    # Consistency: is given if the rules extracted under different training sessions produce the same classifications of test examples
    # Comprehensibility: is determined by measuring the number of rules and the number of antecedents per rule


if __name__ == "__main__":
    main()
