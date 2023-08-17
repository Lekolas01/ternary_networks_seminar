import unittest
from gen_data import *
from bool_formula import *
from train_model import *
from dataloading import FileDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from nn_to_bool_formula import *


class TestFullCircle(unittest.TestCase):
    def full_circle(self, func: Boolean):
        # generate data for function x1 & x2
        data = generate_data(640, func).astype(int)

        # save it into a throwaway folder
        folder_path = Path("unittests/can_delete")
        data_path = folder_path / "logical_AND.csv"
        data.to_csv(data_path, index=False, sep=",")

        # train a neural network on the dataset
        model = nn.Sequential(
            nn.Linear(len(func.all_literals()), 1), nn.Sigmoid(), nn.Flatten(0)
        )
        dataset = FileDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        loss_fn = nn.BCELoss()
        optim = torch.optim.Adam(model.parameters(), 0.01)
        losses = training_loop(model, loss_fn, optim, dataloader, dataloader, epochs=20)

        # convert the trained neural network to a set of perceptrons
        neurons = NeuronNetwork(model, ["x1", "x2"])

        # transform the output perceptron to a boolean function
        found_func = neurons.get_leaf_neuron().to_bool()

        # return the found boolean function
        return found_func

    def test_AND(self):
        target_func = AND([Literal("x1"), Literal("x2")])
        # assert that this boolean function is indeed (x1 & x2)
        found_func = self.full_circle(target_func)
        assert (
            target_func == found_func
        ), f"Did not produce an equivalent function: func: {str(target_func)}, found: {str(found_func)}"
