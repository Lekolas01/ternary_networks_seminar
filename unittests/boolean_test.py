import unittest

from bool_formula import AND, OR, NOT, Literal
from neuron import Neuron, InputNeuron, NeuronGraph
from nn_to_bool_formula import BooleanGraph


class SomethingTest(unittest.TestCase):
    def test_XOR(self):
        x1 = InputNeuron("x1")
        x2 = InputNeuron("x2")
        x3 = Neuron("x3", [(x1, 1.5), (x2, -1.2)], -1.0)
        x4 = Neuron("x4", [(x1, -1.5), (x2, +1.2)], -1.0)
        b = Neuron("b", [(x3, 1.5), (x4, 1.2)], -1.0)
        n_graph = NeuronGraph()
        for neuron in [x1, x2, x3, x4, b]:
            n_graph.add(neuron)
        xor_1 = AND(Literal(x1.name), NOT(Literal(x2.name)))
        xor_2 = AND(NOT(Literal(x1.name)), Literal(x2.name))
        xor = OR(xor_1, xor_2)

        assert BooleanGraph(n_graph) == xor

    def test_XOR_2(self):
        neurons = NeuronGraph()
        x1 = InputNeuron("x1")
        x2 = InputNeuron("x2")
        x3 = Neuron("x3", [(x1, 3.1), (x2, -2.7)], -1.4)
        x4 = Neuron("x4", [(x1, -4.1), (x2, 3.2)], -1.0)
        x5 = Neuron("x5", [(x1, 3.3), (x2, 3.0)], -0.6)
        x6 = Neuron("x6", [(x1, 2.8), (x2, 3.7)], 1.3)
        target = Neuron("target", [(x3, 2.8), (x4, 3.7), (x5, 0.3), (x6, 3.0)], -5.2)
        for n in [x1, x2, x3, x4, x5, x6, target]:
            neurons.add(n)
        xor_1 = AND(Literal(x1.name), NOT(Literal(x2.name)))
        xor_2 = AND(NOT(Literal(x1.name)), Literal(x2.name))
        xor = OR(xor_1, xor_2)
        assert BooleanGraph(neurons) == xor
