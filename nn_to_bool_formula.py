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
        return f"{self.name:>8}  :=  {right_term}"

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

        # sort neurons by their weight
        neurons_in = sorted(self.neurons_in, key=lambda x: abs(x[1]), reverse=True)

        # remember which weights are negative and then invert all of them (needed for negative numbers)
        negative = [tup[1] < 0 for tup in neurons_in]
        neurons_in = [(tup[0], abs(tup[1])) for tup in neurons_in]

        positive_weights = zip(negative, [tup[1] for tup in neurons_in])
        filtered_weights = filter(lambda tup: tup[0], positive_weights)
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
    def __init__(self):
        self.new_neuron_idx = 1  # for naming new neurons
        self.neurons: list[Neuron] = []  # collection of all neurons added to Network
        self.neuron_names: set[str] = set()  # keeps track of the names of all neurons

    def __len__(self):
        return len(self.neurons)

    def __str__(self) -> str:
        return "\n".join(str(neuron) for neuron in self.neurons)

    def add_module(self, net: nn.Module, input_vars: list[str] = []):
        self.input_vars = input_vars  # the names of the input variables
        if not isinstance(net, nn.Sequential):
            raise ValueError("Only allows Sequential for now.")
        first_layer = net[0]
        if not isinstance(first_layer, nn.Linear):
            raise ValueError("First layer must always be a linear layer.")
        shape_out, shape_in = first_layer.weight.shape
        if len(self.input_vars) == 0:
            self.input_vars = [self._new_name() for _ in range(shape_in)]
        if len(self.input_vars) != shape_in:
            raise ValueError("varnames need same shape as input of first layer")

        # create a neuron for each of the input nodes in the first layer
        for idx, name in enumerate(self.input_vars):
            self.add(InputNeuron(name))

        ll_start, ll_end = 0, len(self.neurons)
        for layer in net:
            if isinstance(layer, nn.Linear):
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
        while f"x{self.new_neuron_idx}" in self.neuron_names:
            self.new_neuron_idx += 1
        return f"x{self.new_neuron_idx}"

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


def full_circle(
    target_func: Bool, model: nn.Sequential, epochs=5, verbose=False, seed=None
):
    layer_1 = model[0]
    assert isinstance(layer_1, nn.Linear)
    shape_out, shape_in = layer_1.weight.shape

    # generate data for function
    vars = sorted(list(target_func.all_literals()))
    if shape_in < len(vars):
        raise ValueError(
            f"The input shape of the model is to small, it needs at least {len(vars)}, but got {shape_in}"
        )
    n_dead_vars = shape_in - len(vars)

    for i in range(len(vars) + 1, n_dead_vars + len(vars) + 1):
        dead_var_name = f"x{i}"
        assert dead_var_name not in vars
        vars.append(dead_var_name)
    data = generate_data(640, target_func, vars=vars, seed=seed)

    # save it in a throwaway folder
    folder_path = Path("unittests/can_delete")
    data_path = folder_path / "gen_data.csv"
    data.to_csv(data_path, index=False, sep=",")

    # train a neural network on the dataset
    dataset = FileDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    tracker = Tracker()
    tracker.add_logger(LogMetrics(["timestamp", "epoch", "train_loss", "train_acc"]))
    losses = training_loop(
        model, loss_fn, optim, dataloader, dataloader, epochs=epochs, tracker=tracker
    )

    # convert the trained neural network to a set of perceptrons
    neurons = NeuronGraph()
    neurons.add_module(model, vars)
    if verbose:
        print(neurons)
    # transform the output perceptron to a boolean function
    found_func = BooleanGraph(neurons)

    # return the found boolean function
    return found_func


if __name__ == "__main__":
    args = sys.argv[1:]
    seed = None
    if len(args) > 0:
        seed = int(args[0])
        print(f"{seed = }")
        torch.manual_seed(seed)
        random.seed(seed)

    parity = PARITY(("x1", "x2", "x3"))
    n = len(parity.all_literals())
    model = nn.Sequential(
        nn.Linear(n, n),
        nn.Sigmoid(),
        # nn.Linear(n, n),
        # nn.Sigmoid(),
        nn.Linear(n, 1),
        nn.Sigmoid(),
        nn.Flatten(0),
    )
    found = full_circle(parity, model, epochs=45, seed=seed)
    print(found)
    print(f"{fidelity(found, parity, True) = }")
