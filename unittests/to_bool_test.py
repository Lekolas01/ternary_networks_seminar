import unittest
from nn_to_bool_formula import *
from bool_formula import *

class TestToBool(unittest.TestCase):
    def equivalent(self, n1: Boolean, n2: Boolean) -> bool:
        literals = list(n1.all_literals().union(n2.all_literals()))
        n_literals = len(literals)

        for i in range(2**n_literals):
            interpretation = dict(
                [(l, ((i >> idx) % 2 == 1)) for idx, l in enumerate(literals)]
            )
            if n1(interpretation) != n2(interpretation): return False
        return True
    
    def test_AND(self):
        x1 = Neuron("x1")
        x2 = Neuron("x2")
        b = Neuron("b", [(x1, 1.5), (x2, 1.4)], 2.2)
        assert self.equivalent(b.to_bool(), Func(Op.AND, [Literal(x1.name), Literal(x2.name)]))

    def test_OR(self):
        x1 = Neuron("x1")
        x2 = Neuron("x2")
        b = Neuron("b", [(x1, 1.5), (x2, 1.4)], 1.0)
        assert self.equivalent(b.to_bool(), Func(Op.OR, [Literal(x1.name), Literal(x2.name)]))

    def test_IMPL(self):
        # a => b is the same as !a OR b
        x1 = Neuron("x1")
        x2 = Neuron("x2")
        b = Neuron("b", [(x1, 1.5), (x2, -1.4)], 1.0)
        assert self.equivalent(b.to_bool(), Func(Op.OR, [Literal(x1.name), Literal(x2.name, False)]))

    def test_XOR(self):
        x1 = Neuron("x1")
        x2 = Neuron("x2")
        x3 = Neuron("x3", [(x1, 1.5), (x2, -1.2)], 1.0)
        x4 = Neuron("x4", [(x1, -1.5), (x2, +1.2)], 1.0)
        b = Neuron("b", [(x3, 1.5), (x4, 1.2)], 1.0)
        xor_1 = Func(Op.AND, [Literal(x1.name), Literal(x2.name, False)])
        xor_2 = Func(Op.AND, [Literal(x1.name, False), Literal(x2.name)])
        xor = Func(Op.OR, [xor_1, xor_2])
        assert self.equivalent(b.to_bool(), xor)