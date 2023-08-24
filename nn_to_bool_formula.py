from __future__ import annotations
from bool_formula import *
import torch.nn as nn
from enum import Enum
from bool_formula import Interpretation
from gen_data import *
from dataloading import *
from train_model import *
from torch.utils.data import DataLoader


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
        return f'Neuron("{self.name}")'

    def to_bool(self) -> Boolean:
        def to_bool_rec(
            neurons_in: list[tuple[Neuron, float]],
            neuron_signs: list[bool],
            threshold: float,
            i: int = 0,
        ) -> Boolean:
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
                [
                    to_bool_rec(neurons_in, neuron_signs, threshold - weight, i + 1),
                    Literal(name, positive),
                ]
            )
            return OR([term1, term2])

        # sort neurons by their weight
        neurons_in = sorted(self.neurons_in, key=lambda x: abs(x[1]), reverse=True)

        # remember which weights are negative and then invert all of them (needed for negative numbers)
        negative = [tup[1] < 0 for tup in neurons_in]
        neurons_in = [(tup[0], abs(tup[1])) for tup in neurons_in]

        positive_weights = zip(negative, [tup[1] for tup in neurons_in])
        filtered_weights = filter(lambda tup: tup[0], positive_weights)
        bias_diff = sum(tup[1] for tup in filtered_weights)
        long_ans = to_bool_rec(neurons_in, negative, -self.bias + bias_diff)

        ans = simplified(long_ans)
        return ans


class NeuronNetwork:
    def __init__(self, net: nn.Module, input_vars: list[str] = []):
        self.new_neuron_idx = 1  # for naming new neurons
        self.neurons = []  # collection of all neurons added to Network
        self.input_vars = input_vars  # the names of the input variables
        self.neuron_names = set()  # keeps track of the names of all neurons

        if isinstance(net, nn.Sequential):
            first_layer = net[0]
            if not isinstance(first_layer, nn.Linear):
                raise ValueError("First layer must always be a linear layer.")
            shape_out, shape_in = first_layer.weight.shape
            if len(input_vars) == 0:
                input_vars = [self._new_name() for _ in range(shape_in)]
            if len(input_vars) != shape_in:
                raise ValueError("varnames need same shape as input of first layer")

            # create a neuron for each of the input nodes in the first layer
            for idx, name in enumerate(input_vars):
                self.add_neuron(Neuron(name, neurons_in=[]))

            ll_start, ll_end = 0, len(self.neurons)
            for layer in net:
                if isinstance(layer, nn.Linear):
                    shape_out, shape_in = layer.weight.shape
                    weight = layer.weight.tolist()
                    bias = layer.bias.tolist()

                    for idx in range(shape_out):
                        neurons_in = list(
                            zip(self.neurons[ll_start:ll_end], weight[idx])
                        )
                        name = self._new_name()
                        neuron = Neuron(
                            name,
                            neurons_in=neurons_in,
                            bias=bias[idx],
                        )
                        self.add_neuron(neuron)
                    ll_start, ll_end = ll_end, len(self.neurons)

            # rename the last variable, so it is distinguishable from the rest
            self.rename_neuron(self.target_neuron(), "target")

        else:
            raise ValueError("Only allows Sequential for now.")

    def __len__(self):
        return len(self.neurons)

    def __str__(self) -> str:
        return "\n".join(str(neuron) for neuron in self.neurons)

    def add_neuron(self, neuron: Neuron):
        assert neuron.name not in self.neuron_names
        self.neurons.append(neuron)
        self.neuron_names.add(neuron.name)

    def rename_neuron(self, neuron: Neuron, new_name: str):
        assert neuron in self.neurons
        assert new_name not in self.neuron_names
        neuron.name = new_name

    def _new_name(self):
        while f"x{self.new_neuron_idx}" in self.neuron_names:
            self.new_neuron_idx += 1
        return f"x{self.new_neuron_idx}"

    def target_neuron(self) -> Neuron:
        return self.neurons[-1]


def full_circle(target_func: Boolean, model: nn.Sequential, epochs=5):
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

    for i in range(n_dead_vars):
        dead_var_name = f"dead{i}"
        assert dead_var_name not in vars
        vars.append(dead_var_name)
    data = generate_data(640, target_func, vars=vars)

    # save it in a throwaway folder
    folder_path = Path("unittests/can_delete")
    data_path = folder_path / "gen_data.csv"
    data.to_csv(data_path, index=False, sep=",")

    # train a neural network on the dataset
    dataset = FileDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss_fn = nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), 0.01)
    losses = training_loop(model, loss_fn, optim, dataloader, dataloader, epochs=epochs)

    # convert the trained neural network to a set of perceptrons
    neurons = NeuronNetwork(model, input_vars=vars)

    # transform the output perceptron to a boolean function
    found_func = neurons.target_neuron().to_bool()
    print(neurons)
    # return the found boolean function
    return found_func


if __name__ == "__main__":
    target_funcs = [
        OR(
            [
                AND([Literal("x1", False), Literal("x2", True)]),
                AND([Literal("x1", True), Literal("x2", False)]),
            ]
        ),
        OR(
            [
                AND([Literal("x1", False), Literal("x2", False)]),
                AND([Literal("x1", True), Literal("x2", True)]),
            ]
        ),
    ]
    for target_func in target_funcs:
        model = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
            nn.Flatten(0),
        )
        found_func = full_circle(target_func, model)
        assert (
            target_func == found_func
        ), f"Did not produce an equivalent function: {target_func = }; {found_func = }"
