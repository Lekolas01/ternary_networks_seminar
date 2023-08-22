import unittest
from gen_data import *
from bool_formula import *
from train_model import *
from dataloading import FileDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from nn_to_bool_formula import *


class TestFullCircle(unittest.TestCase):
    def full_circle(self, target_func: Boolean, model: nn.Sequential):
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
        data = generate_data(640, target_func, vars=vars).astype(int)

        # save it in a throwaway folder
        folder_path = Path("unittests/can_delete")
        data_path = folder_path / "gen_data.csv"
        data.to_csv(data_path, index=False, sep=",")

        # train a neural network on the dataset
        dataset = FileDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        loss_fn = nn.BCELoss()
        optim = torch.optim.Adam(model.parameters(), 0.01)
        losses = training_loop(model, loss_fn, optim, dataloader, dataloader, epochs=2)

        # convert the trained neural network to a set of perceptrons
        neurons = NeuronNetwork(model, varnames=vars)

        # transform the output perceptron to a boolean function
        found_func = neurons.get_leaf_neuron().to_bool()

        # return the found boolean function
        return found_func

    def test_binaryFunctions(self):
        target_funcs = [
            Constant(False),
            Constant(True),
            Literal("x1", False),
            Literal("x1", True),
            AND([Literal("x1", False), Literal("x2", False)]),
            AND([Literal("x1", False), Literal("x2", True)]),
            AND([Literal("x1", True), Literal("x2", False)]),
            AND([Literal("x1", True), Literal("x2", True)]),
            OR([Literal("x1", False), Literal("x2", False)]),
            OR([Literal("x1", False), Literal("x2", True)]),
            OR([Literal("x1", True), Literal("x2", False)]),
            OR([Literal("x1", True), Literal("x2", True)]),
        ]
        for target_func in target_funcs:
            model = nn.Sequential(
                nn.Linear(12, 1),
                nn.Sigmoid(),
                nn.Flatten(0),
            )
            found = self.full_circle(target_func, model)
            assert (
                target_func == found
            ), f"Did not produce an equivalent function: {target_func = }; {found = }"

    def test_XOR(self):
        torch.manual_seed(0)
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
            found_func = self.full_circle(target_func, model)
            assert (
                target_func == found_func
            ), f"Did not produce an equivalent function: {target_func = }; {found_func = }"
