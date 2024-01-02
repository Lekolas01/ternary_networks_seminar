from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bool_formula import Bool, Interpretation
from datasets import FileDataset
from graphics import plot_neuron_dist
from neuron import InputNeuron, Neuron2, NeuronGraph2
from node import Graph


class BooleanGraph(Bool):
    def __init__(self, ng: NeuronGraph2) -> None:
        super().__init__()
        self.ng = ng
        self.neurons: list[Neuron2] = []
        self.neuron_names: list[str] = []
        self.bools: list[Bool] = []
        self.input_neurons = set()
        for neuron in self.ng.neurons:
            print(neuron)
            self.neurons.append(neuron)
            self.neuron_names.append(neuron.name)
            n_bool = neuron.to_bool()
            print(n_bool)
            print()
            self.bools.append(n_bool)
            if not isinstance(neuron, InputNeuron):
                pass
                plot_neuron_dist(neuron)
            else:
                self.input_neurons.add(neuron)

        self.name_2_idx = {name: i for i, name in enumerate(self.neuron_names)}
        self.target_neuron = self.ng.target()
        # simplify the boolean graph
        # self.simplify()

    def __call__(self, interpretation: Interpretation) -> bool:
        int_copy = copy.copy(interpretation)
        for i, neuron in enumerate(self.neurons):
            n_bool = self.bools[i]
            int_copy[neuron.name] = n_bool(int_copy)

        target_name = self.ng.target().name
        return int_copy[target_name]

    def __getitem__(self, key: str) -> Bool:
        return self.bools[self.name_2_idx[key]]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        ans = "BooleanGraph[\n"
        for i, n_bool in enumerate(self.bools):
            name = self.neuron_names[i]
            ans += f"\t{name} := {str(n_bool)}\n"
        ans += "]\n"
        return ans

    def all_literals(self) -> set[str]:
        ans = set()
        for n_bool in self.bools:
            ans = ans.union(n_bool.all_literals())
        return ans

    def negated(self) -> Bool:
        raise NotImplementedError

    def simplify(self) -> None:
        n = len(self.bools)
        usages = np.zeros((n, n), dtype=np.int8)
        node_exists = np.ones(n, dtype=np.int8)
        for idx, n_bool in enumerate(self.bools):
            for other_name in n_bool.all_literals():
                other_idx = self.name_2_idx[other_name]
                usages[idx][other_idx] = 1

        # delete all nodes that have 0 input- or output nodes
        # repeat this until no nodes were deleted
        deleted_nodes = True
        while deleted_nodes:
            deleted_nodes = False
            for i in range(n):
                if (
                    not node_exists[i]
                    or self.neuron_names[i] == self.target_neuron.name
                ):
                    continue
                if sum(usages[i]) == 0 or sum(usages[:, i]) == 0:
                    node_exists[i] = False
                    usages[i] = 0
                    usages[:, i] = 0

        # delete nodes that only have one input node and re-connect their input and output nodes
        for idx, n_bool in [
            (i, n_bool) for i, n_bool in enumerate(self.bools) if node_exists[i]
        ]:
            # input_nodes = n_bool.all_literals()
            # if len(input_nodes) != 1 or node in self.input_names:
            #    continue
            pass

        # delete all nodes that were marked for deletion
        for i in range(node_exists.shape[0]):
            if not node_exists[i]:
                del self.neurons[i]
                del self.bools[i]
                del self.neuron_names[i]
                del self.neuron_names[i]


def fidelity(
    model1: nn.Sequential, bg: BooleanGraph, data_path: Path
) -> tuple[float, float]:
    dl = DataLoader(FileDataset(data_path))
    fid = 0
    n_vals = 0
    bg_accuracy = 0
    df = pd.read_csv(data_path)
    vars = list(df.columns[:-1])

    for X, y in dl:
        nn_pred = model1(X)
        nn_pred = [bool(val.round()) for val in nn_pred]

        data = []
        n_rows, n_cols = X.shape
        for i in range(n_rows):
            row = X[i]
            new_var = {vars[j]: bool(row[j]) for j in range(n_cols)}
            data.append(new_var)
        bg_pred = [bg(datapoint) for datapoint in data]
        bg_correct = [bg_pred[i] == y[i] for i in range(n_rows)]
        n_correct = len(list(filter(lambda x: x, bg_correct)))
        bg_accuracy += n_correct
        bg_same = [bg_pred[i] == nn_pred[i] for i in range(n_rows)]
        n_same = len(list(filter(lambda x: x, bg_same)))
        fid += n_same
        n_vals += n_rows
    return fid / n_vals, bg_accuracy / n_vals


def nn_to_rule_set(model_path: Path, data_path: Path) -> BooleanGraph:
    assert os.path.isfile(model_path)
    assert os.path.isfile(data_path)
    model = torch.load(model_path)

    df = pd.read_csv(data_path)
    dl = DataLoader(FileDataset(data_path))

    vars = list(df.columns[:-1])
    # transform the trained neural network to a directed graph of perceptrons
    neuron_graph = NeuronGraph2(vars, model)
    # transform the neuron graph to a boolean function
    bool_graph = BooleanGraph(neuron_graph)
    return bool_graph


def main():
    # Fidelity: the percentage of test examples for which the classification made by the rules agrees with the neural network counterpart
    # Accuracy: the percentage of test examples that are correctly classified by the rules
    # Consistency: is given if the rules extracted under different training sessions produce the same classifications of test examples
    # Comprehensibility: is determined by measuring the number of rules and the number of antecedents per rule
    # fid, rules_acc = fidelity(vars, model, rules, df)
    # nn_acc = acc(model, train_dl, torch.device("cpu"))
    # print(f"{nn_acc = }")
    # print(f"{fid = }")
    # print(f"{rules_acc = }")
    pass


if __name__ == "__main__":
    main()
