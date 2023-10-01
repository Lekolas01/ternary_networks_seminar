import unittest
from gen_data import *
from bool_formula import *
import torch.nn as nn
from nn_to_bool_formula import full_circle


class TestFullCircle(unittest.TestCase):
    def test_binaryFunctions(self):
        target_funcs = [
            Constant(False),
            Constant(True),
            NOT(Literal("x1")),
            Literal("x1"),
            AND(NOT(Literal("x1")), NOT(Literal("x2"))),
            AND(NOT(Literal("x1")), Literal("x2")),
            AND(Literal("x1"), NOT(Literal("x2"))),
            AND(Literal("x1"), Literal("x2")),
            OR(NOT(Literal("x1")), NOT(Literal("x2"))),
            OR(NOT(Literal("x1")), Literal("x2")),
            OR(Literal("x1"), NOT(Literal("x2"))),
            OR(Literal("x1"), Literal("x2")),
        ]
        for target_func in target_funcs:
            n_vars = len(target_func.all_literals())
            model = nn.Sequential(
                nn.Linear(n_vars, 1),
                nn.Sigmoid(),
                nn.Flatten(0),
            )
            found = full_circle(target_func, model, epochs=10)["bool_graph"]
            assert (
                target_func == found
            ), f"Did not produce an equivalent function: {target_func = }; {found = }"

    def test_XOR(self):
        target_funcs = [
            OR(
                AND(NOT("x1"), "x2"),
                AND(Literal("x1"), NOT(Literal("x2"))),
            ),
            OR(
                AND(NOT(Literal("x1")), NOT(Literal("x2"))),
                AND(Literal("x1"), Literal("x2")),
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
            ans = full_circle(target_func, model, epochs=80)
            found_func = ans['bool_graph']
            names = target_func.all_literals().union(found_func.all_literals())
            data = all_interpretations(names)
            assert (
                fidelity(target_func, found_func, data) >= 0.8
            ), f"Did not produce an equivalent function: {target_func = }; {found_func = }"
            break
