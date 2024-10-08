import unittest
from math import isclose

import numpy as np
import torch
import torch.nn as nn

from bool_formula import AND, NOT, OR, Bool, Constant, Literal
from neuron import (
    Neuron,
    NeuronGraph,
    QuantizedNeuron,
    QuantizedNeuronGraph,
    RuleSetGraph,
    to_vars,
)
from node import Graph


class TestToBool(unittest.TestCase):
    def test_NeuronGraphs(self):
        n = 5
        keys = [f"a{i + 1}" for i in range(n)]
        model = nn.Sequential(
            nn.Linear(n, n + 1),
            nn.Tanh(),
            nn.Linear(n + 1, n - 1),
            nn.Tanh(),
            nn.Linear(n - 1, 1),
            nn.Sigmoid(),
            nn.Flatten(0),
        ).train()
        neuron_graph = NeuronGraph(model, keys)
        random_x = torch.rand(n)
        random_input_vars = to_vars(random_x, keys)
        assert isclose(
            model(random_x).float(), neuron_graph(random_input_vars), rel_tol=1e-7
        )

    def test_False(self):
        x1 = Neuron2("x1")
        x2 = Neuron2("x2")
        b = Neuron2("b", [(x1, 1.5), (x2, 1.4)], -5.0)
        assert b.to_bool() == Constant(False)

    def test_NegativeWeights(self):
        x1 = Neuron2("x1")
        b = Neuron2("b", [(x1, -1)], -0.5)
        b2 = Neuron2("b2", [(x1, -1)], 0.5)
        assert b.to_bool() == Constant(False)
        assert b2.to_bool() == NOT(Literal(x1.name))

    def test_all_binary_Funcs(self):
        x1 = Neuron2("x1")
        x2 = Neuron2("x2")
        b1 = Neuron2("b1", [(x1, 1), (x2, 1)], -1.5)  # x1 AND x2
        b2 = Neuron2("b4", [(x1, 1), (x2, 1)], -0.5)  # x1 OR x2
        b3 = Neuron2("b2", [(x1, 1), (x2, -1)], -0.5)  # x1 AND !x2
        b4 = Neuron2("b5", [(x1, 1), (x2, -1)], 0.5)  # x1 OR !x2
        b5 = Neuron2("b3", [(x1, -1), (x2, -1)], 0.5)  # !x1 AND !x2
        b6 = Neuron2("b6", [(x1, -1), (x2, -1)], 1.5)  # !x1 OR !x2
        assert b1.to_bool() == AND(Literal(x1.name), Literal(x2.name))
        assert b2.to_bool() == OR(Literal(x1.name), Literal(x2.name))
        assert b3.to_bool() == AND(Literal(x1.name), NOT(Literal(x2.name)))
        assert b4.to_bool() == OR(Literal(x1.name), NOT(Literal(x2.name)))
        assert b5.to_bool() == AND(NOT(Literal(x1.name)), NOT(Literal(x2.name)))
        assert b6.to_bool() == OR(NOT(Literal(x1.name)), NOT(Literal(x2.name)))

    def test_ComplexNeuron(self):
        x1 = Neuron2("x1")
        x2 = Neuron2("x2")
        x3 = Neuron2("x3")
        x4 = Neuron2("x4")
        b = Neuron2("b", [(x1, 1.5), (x2, -1.4), (x3, 2.1), (x4, -0.3)], -1.0)
        true_bool = OR(
            AND(Literal(x1.name), NOT(Literal(x2.name))),
            AND(Literal(x3.name), OR(Literal(x1.name), NOT(Literal(x2.name)))),
        )
        assert b.to_bool() == true_bool

    def test_simplified(self):
        formulae: list[tuple[Bool, str]] = [
            (AND(), "T"),
            (AND(True), "T"),
            (AND(False), "F"),
            (AND(True, True), "T"),
            (AND(True, False), "F"),
            (AND(False, False), "F"),
            (AND(True, True, "x1"), "x1"),
            (AND(True, False, "x1"), "F"),
            (AND(False, False, "x1"), "F"),
            (NOT(True), "F"),
            (NOT(NOT(True)), "T"),
            (NOT(False), "T"),
            (NOT(NOT(False)), "F"),
            (NOT("x1"), "!x1"),
            (NOT(NOT("x1")), "x1"),
            (OR(), "F"),
            (NOT(AND()), "F"),
            (NOT(AND("x1")), "!x1"),
            (NOT(AND("x1", "x2")), "(!x1 | !x2)"),
            (AND(AND(True, "x1"), True), "x1"),
            (AND(AND(True, NOT("x1")), True), "!x1"),
            (NOT(AND(AND(True, NOT("x1")), True)), "x1"),
        ]

        for formula, ans in formulae:
            if formula.simplified().__str__() != ans:
                assert False

    def test_XOR(self):
        x1 = InputNeuron("x1")
        x2 = InputNeuron("x2")
        x3 = Neuron2("x3", [(x1, 1.5), (x2, -1.2)], -1.0)
        x4 = Neuron2("x4", [(x1, -1.5), (x2, +1.2)], -1.0)
        b = Neuron2("b", [(x3, 1.5), (x4, 1.2)], -1.0)
        n_graph = NeuronGraph2()
        for neuron in [x1, x2, x3, x4, b]:
            n_graph.add(neuron)
        xor_1 = AND(Literal(x1.name), NOT(Literal(x2.name)))
        xor_2 = AND(NOT(Literal(x1.name)), Literal(x2.name))
        xor = OR(xor_1, xor_2)

        assert BoolGraph(n_graph) == xor

    def test_XOR_2(self):
        neurons = NeuronGraph2()
        x1 = InputNeuron("x1")
        x2 = InputNeuron("x2")
        x3 = Neuron2("x3", [(x1, 3.1), (x2, -2.7)], -1.4)
        x4 = Neuron2("x4", [(x1, -4.1), (x2, 3.2)], -1.0)
        x5 = Neuron2("x5", [(x1, 3.3), (x2, 3.0)], -0.6)
        x6 = Neuron2("x6", [(x1, 2.8), (x2, 3.7)], 1.3)
        target = Neuron2("target", [(x3, 2.8), (x4, 3.7), (x5, 0.3), (x6, 3.0)], -5.2)
        for n in [x1, x2, x3, x4, x5, x6, target]:
            neurons.add(n)
        xor_1 = AND(Literal(x1.name), NOT(Literal(x2.name)))
        xor_2 = AND(NOT(Literal(x1.name)), Literal(x2.name))
        xor = OR(xor_1, xor_2)
        assert BoolGraph(neurons) == xor

    def test_random_model(self):
        n = 5
        keys = [f"x{i + 1}" for i in range(n)]
        model = nn.Sequential(
            nn.Linear(n, n + 1, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(n + 1, n - 1, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(n - 1, 1, dtype=torch.float64),
            nn.Sigmoid(),
            nn.Flatten(0),
        ).train(False)
        neurons = to_neurons(model, keys)
        neuron_graph = Graph(neurons)
        random_x = torch.rand(n, dtype=torch.float64)
        random_input_vars = to_vars(random_x, keys)
        print(f"{model(random_x).item() = }")
        print(f"{neuron_graph(random_input_vars) = }")
        assert isclose(
            model(random_x).float(), neuron_graph(random_input_vars), rel_tol=1e-6
        )
