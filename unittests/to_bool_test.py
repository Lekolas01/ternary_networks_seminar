import unittest
from nn_to_bool_formula import *
from bool_formula import *


class TestToBool(unittest.TestCase):
    def test_True(self):
        x1 = Neuron("x1")
        x2 = Neuron("x2")
        b = Neuron("b", [(x1, 1.5), (x2, 1.4)], 2.2)
        assert b.to_bool() == Constant(True)

    def test_False(self):
        x1 = Neuron("x1")
        x2 = Neuron("x2")
        b = Neuron("b", [(x1, 1.5), (x2, 1.4)], -5.0)
        assert b.to_bool() == Constant(False)

    def test_NegativeWeights(self):
        x1 = Neuron("x1")
        b = Neuron("b", [(x1, -1)], -0.5)
        b2 = Neuron("b2", [(x1, -1)], 0.5)
        assert b.to_bool() == Constant(False)
        assert b2.to_bool() == Literal(x1.name, False)

    def test_all_binary_Funcs(self):
        x1 = Neuron("x1")
        x2 = Neuron("x2")
        b1 = Neuron("b1", [(x1, 1), (x2, 1)], -1.5)  # x1 AND x2
        b2 = Neuron("b4", [(x1, 1), (x2, 1)], -0.5)  # x1 OR x2
        b3 = Neuron("b2", [(x1, 1), (x2, -1)], -0.5)  # x1 AND !x2
        b4 = Neuron("b5", [(x1, 1), (x2, -1)], 0.5)  # x1 OR !x2
        b5 = Neuron("b3", [(x1, -1), (x2, -1)], 0.5)  # !x1 AND !x2
        b6 = Neuron("b6", [(x1, -1), (x2, -1)], 1.5)  # !x1 OR !x2
        assert b1.to_bool() == AND([Literal(x1.name), Literal(x2.name)])
        assert b2.to_bool() == OR([Literal(x1.name), Literal(x2.name)])
        assert b3.to_bool() == AND([Literal(x1.name), Literal(x2.name, False)])
        assert b4.to_bool() == OR([Literal(x1.name), Literal(x2.name, False)])
        assert b5.to_bool() == AND([Literal(x1.name, False), Literal(x2.name, False)])
        assert b6.to_bool() == OR([Literal(x1.name, False), Literal(x2.name, False)])

    def test_ComplexNeuron(self):
        x1 = Neuron("x1")
        x2 = Neuron("x2")
        x3 = Neuron("x3")
        x4 = Neuron("x4")
        b = Neuron("b", [(x1, 1.5), (x2, -1.4), (x3, 2.1), (x4, -0.3)], -1.0)
        true_bool = OR(
            [
                AND([Literal(x1.name), Literal(x2.name, False)]),
                AND(
                    [Literal(x3.name), OR([Literal(x1.name), Literal(x2.name, False)])]
                ),
            ]
        )
        assert b.to_bool() == true_bool

    def test_TanhActivation(self):
        pass

    def test_XOR(self):
        x1 = Neuron("x1")
        x2 = Neuron("x2")
        x3 = Neuron("x3", [(x1, 1.5), (x2, -1.2)], 1.0)
        x4 = Neuron("x4", [(x1, -1.5), (x2, +1.2)], 1.0)
        b = Neuron("b", [(x3, 1.5), (x4, 1.2)], 1.0)
        xor_1 = AND([Literal(x1.name), Literal(x2.name, False)])
        xor_2 = AND([Literal(x1.name, False), Literal(x2.name)])
        xor = OR([xor_1, xor_2])
        assert self.equivalent(b.to_bool(), xor)
