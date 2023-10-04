from __future__ import annotations
from bool_formula import *
import torch.nn as nn
from enum import Enum
from bool_formula import Bool, Interpretation
from gen_data import *
from dataloading import *
from train_model import *
from torch.utils.data import DataLoader
from loggers.loggers import *
import sys
from typing import Any, Iterable
from utils import accuracy
import matplotlib.pyplot as plt


class Act(Enum):
    SIGMOID = 1
    TANH = 2


class Neuron:
    def __init__(
        self,
        name: str,
        neurons_in: list[tuple[Neuron, float]] = [],
        bias: float = 0.0,
        activation_in: Act = Act.SIGMOID,
    ) -> None:
        self.name = name
        self.neurons_in = neurons_in
        self.bias = bias
        self.activation_in = activation_in

    def __str__(self) -> str:
        right_term = ""
        act_str = "sig" if self.activation_in == Act.SIGMOID else "tanh"

        if len(self.neurons_in) >= 1:
            neuron, weight = self.neurons_in[0]
            right_term += f"{round(weight, 2):>5}*{neuron.name:<5}"
        for neuron, weight in self.neurons_in[1:]:
            weight_sign = "+" if weight >= 0 else "-"
            right_term += f"{weight_sign}{round(abs(weight), 2):>5}*{neuron.name:<5}"

        if self.bias:
            weight_sign = "+" if self.bias >= 0 else "-"
            right_term += f"{weight_sign}{round(abs(self.bias), 2)}"
        if not right_term:
            right_term = self.name
        return f"{self.name:>8}  :=  {act_str}({right_term})"

    def __repr__(self) -> str:
        return f'Neuron("{str(self)}")'

    def to_bool(self) -> Bool:
        def to_bool_rec(
            neurons_in: list[tuple[Neuron, float]],
            neuron_signs: list[bool],
            threshold: float,
            i: int = 0,
        ) -> Bool:
            if threshold < 0:
                return Constant(True)
            if i == len(neurons_in):
                return Constant(False)

            name = neurons_in[i][0].name
            weight = neurons_in[i][1]
            positive = not neuron_signs[i]

            # set to False
            term1 = to_bool_rec(neurons_in, neuron_signs, threshold, i + 1)
            term2 = AND(
                to_bool_rec(neurons_in, neuron_signs, threshold - weight, i + 1),
                Literal(name) if positive else NOT(Literal(name)),
            )

            return OR(term1, term2)

        for idx, (neuron_in, weight) in enumerate(self.neurons_in):
            if (
                not isinstance(neuron_in, InputNeuron)
                and neuron_in.activation_in == Act.TANH
            ):
                # a = -1
                self.bias -= 1

                # k = 2
                a = self.neurons_in[idx]
                a = list(a)
                a[1] *= 2
                self.neurons_in[idx] = tuple(a)

        # sort neurons by their weight
        neurons_in = sorted(self.neurons_in, key=lambda x: abs(x[1]), reverse=True)

        # remember which weights are negative and then invert all of them (needed for negative numbers)
        negative = [tup[1] < 0 for tup in neurons_in]
        neurons_in = [(tup[0], abs(tup[1])) for tup in neurons_in]

        positive_weights = list(zip(negative, [tup[1] for tup in neurons_in]))
        filtered_weights = list(filter(lambda tup: tup[0], positive_weights))
        bias_diff = sum(tup[1] for tup in filtered_weights)
        return to_bool_rec(neurons_in, negative, -self.bias + bias_diff).simplified()


class InputNeuron(Neuron):
    def __init__(self, name: str) -> None:
        self.name = name

    def to_bool(self) -> Bool:
        return Literal(self.name)

    def __str__(self) -> str:
        return f'InputNeuron("{self.name}")'


class NeuronGraph:
    def __init__(
        self, vars: Optional[list[str]] = None, net: Optional[nn.Module] = None
    ):
        self.new_neuron_idx = 1  # for naming new neurons
        self.neurons: list[Neuron] = []  # collection of all neurons added to Network
        self.neuron_names: set[str] = set()  # keeps track of the names of all neurons
        if net:
            assert vars is not None
            self.add_module(net, vars)

    def __len__(self):
        return len(self.neurons)

    def __str__(self) -> str:
        return "\n".join(str(neuron) for neuron in self.neurons)

    def add_module(self, net: nn.Module, input_vars: list[str]):
        self.input_vars = input_vars  # the names of the input variables
        if not isinstance(net, nn.Sequential):
            raise ValueError("Only allows Sequential for now.")
        first_layer = net[0]
        if not isinstance(first_layer, nn.Linear):
            raise ValueError("First layer must always be a linear layer.")
        shape_out, shape_in = first_layer.weight.shape
        if len(self.input_vars) != shape_in:
            raise ValueError("varnames need same shape as input of first layer")

        # create a neuron for each of the input nodes in the first layer
        for idx, name in enumerate(self.input_vars):
            self.add(InputNeuron(name))

        ll_start, ll_end = 0, len(self.neurons)
        curr_act = Act.SIGMOID
        for idx, layer in enumerate(net):
            if isinstance(layer, nn.Linear):
                next_layer = net[idx + 1]
                if isinstance(next_layer, nn.Sigmoid):
                    curr_act = Act.SIGMOID
                elif isinstance(next_layer, nn.Tanh):
                    curr_act = Act.TANH

                shape_out, shape_in = layer.weight.shape
                weight = layer.weight.tolist()
                bias = layer.bias.tolist()

                for idx in range(shape_out):
                    neurons_in = list(zip(self.neurons[ll_start:ll_end], weight[idx]))
                    name = self._new_name()
                    neuron = Neuron(
                        name,
                        neurons_in=neurons_in,
                        bias=bias[idx],
                        activation_in=curr_act,
                    )
                    self.add(neuron)
                ll_start, ll_end = ll_end, len(self.neurons)

        # rename the last variable, so it is distinguishable from the rest
        self.rename(self.target(), "target")

    def add(self, neuron: Neuron):
        assert neuron.name not in self.neuron_names
        self.neurons.append(neuron)
        self.neuron_names.add(neuron.name)

    def rename(self, neuron: Neuron, new_name: str):
        assert neuron in self.neurons
        assert new_name not in self.neuron_names
        neuron.name = new_name

    def _new_name(self):
        while f"h{self.new_neuron_idx}" in self.neuron_names:
            self.new_neuron_idx += 1
        return f"h{self.new_neuron_idx}"

    def target(self) -> Neuron:
        return self.neurons[-1]


class BooleanGraph(Bool):
    def __init__(self, neurons: NeuronGraph) -> None:
        super().__init__()
        self.neurons = neurons
        self.bools = {n.name: n.to_bool() for n in self.neurons.neurons}

    def __call__(self, interpretation: Interpretation) -> bool:
        int_copy = copy.copy(interpretation)
        for key in self.bools:
            n_bool = self.bools[key]
            val = n_bool(int_copy)

            int_copy[key] = val

        target_name = self.neurons.target().name
        return int_copy[target_name]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        ans = "BooleanGraph[\n"
        for key in self.bools:
            n_bool = self.bools[key]
            ans += f"\t{key} := {str(n_bool)}\n"
        ans += "]\n"
        return ans

    def all_literals(self) -> set[str]:
        ans = set()
        for key in self.bools:
            if isinstance(self.bools[key], InputNeuron):
                ans = ans.union(self.bools[key].all_literals())
        return ans

    def negated(self) -> Bool:
        raise NotImplementedError


class Multiply(nn.Module):
    def __init__(self):
        super(Multiply, self).__init__()

    def forward(self, x):
        return x * 4


def full_circle(
    target_func: Bool,
    model: nn.Sequential,
    n_datapoints=4000,
    epochs=5,
    seed=None,
    verbose=False,
) -> Dict[str, Any]:
    layer_1 = model[0]
    assert isinstance(layer_1, nn.Linear)
    shape_out, shape_in = layer_1.weight.shape

    # generate data for function
    vars = sorted(list(target_func.all_literals()))
    if shape_in != len(vars):
        raise ValueError(
            f"The model's input shape must be same as the number of variables in target_func, but got: {len(vars) = },  {shape_in = }"
        )

    data = generate_data(n_datapoints, target_func, vars=vars, seed=seed)

    # save it in a throwaway folder
    folder_path = Path("unittests/can_delete")
    data_path = folder_path / "gen_data.csv"
    data.to_csv(data_path, index=False, sep=",")

    # train a neural network on the dataset
    dataset = FileDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    tracker = Tracker()
    if verbose:
        tracker.add_logger(
            LogMetrics(["timestamp", "epoch", "train_loss", "train_acc"])
        )
    losses = training_loop(
        model, loss_fn, optim, dataloader, dataloader, epochs=epochs, tracker=tracker
    )
    model.train(False)

    # transform the trained neural network to a directed graph of perceptrons
    neuron_graph = NeuronGraph(vars, model)
    # transform the output perceptron to a boolean function
    bool_graph = BooleanGraph(neuron_graph)

    # return the found boolean function
    return {
        "vars": vars,
        "neural_network": model,
        "losses": losses,
        "dataloader": dataloader,
        "neuron_graph": neuron_graph,
        "bool_graph": bool_graph,
    }


def fidelity(vars, model, bg, dl):
    fid = 0
    n_vals = 0
    bg_accuracy = 0
    for X, y in dl:
        nn_pred = model(X)
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


def test_model(seed, target_func, model):
    ans = full_circle(target_func, model, epochs=200, seed=seed, verbose=False)
    bg = ans["bool_graph"]
    dl = ans["dataloader"]
    vars = ans["vars"]

    fid, bg_acc = fidelity(vars, model, bg, dl)
    nn_acc = accuracy(model, dl, torch.device("cpu"))
    return fid, bg_acc, nn_acc


if __name__ == "__main__":
    # set seed to some integer if you want determinism during training
    seed: Optional[int] = 86704648622300
    # 32697229636700

    if seed is None:
        seed = torch.random.initial_seed()
    else:
        torch.manual_seed(seed)
    random.seed(seed)
    print(f"{seed = }")

    vars = [f"x{i + 1}" for i in range(6)]
    parity = PARITY(vars)
    n = len(parity.all_literals())

    activation_sig = []
    activation_tanh = []

    for i in range(12):
        print(f"\t------ {i} ------")
        for act in ["sigmoid"]:
            print(f"-------------- {act} -----------")
            if act == "sigmoid":
                model = nn.Sequential(
                    nn.Linear(n, n),
                    nn.Sigmoid(),
                    nn.Linear(n, n),
                    nn.Sigmoid(),
                    nn.Linear(n, 1),
                    nn.Sigmoid(),
                    nn.Flatten(0),
                )
            else:
                model = nn.Sequential(
                    nn.Linear(n, n),
                    nn.Tanh(),
                    nn.Linear(n, n),
                    nn.Tanh(),
                    nn.Linear(n, 1),
                    nn.Sigmoid(),
                    nn.Flatten(0),
                )

            fid, bg_acc, nn_acc = test_model(seed, parity, model)
            # Fidelity: the percentage of test examples for which the classification made by the rules agrees with the neural network counterpart
            # Accuracy: the percentage of test examples that are correctly classified by the rules
            # Consistency: is given if the rules extracted under different training sessions produce the same classifications of test examples
            # Comprehensibility: is determined by measuring the number of rules and the number of antecedents per rule
            if act == "sigmoid":
                activation_sig.append((nn_acc, bg_acc, fid))
            else:
                activation_tanh.append((nn_acc, bg_acc, fid))

    sig_nn_acc = [val[0] for val in activation_sig]
    sig_bg_acc = [val[1] for val in activation_sig]
    sig_fid = [val[2] for val in activation_sig]

    tanh_nn_acc = [val[0] for val in activation_tanh]
    tanh_bg_acc = [val[1] for val in activation_tanh]
    tanh_fid = [val[2] for val in activation_tanh]
    print(f"{sig_nn_acc = }")
    print(f"{sig_bg_acc = }")
    print(f"{sig_fid = }")
    print(f"{tanh_nn_acc = }")
    print(f"{tanh_bg_acc = }")
    print(f"{tanh_fid = }")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sig_nn_acc, sig_nn_acc, sig_fid, c="r", label="Sigmoid")
    ax.scatter(tanh_nn_acc, tanh_nn_acc, tanh_fid, c="g", label="Tanh")
    ax.set_xlabel("NN Accuracy")
    ax.set_ylabel("BG Accuracy")
    ax.set_zlabel("Fidelity")
    plt.legend()
    plt.show()
