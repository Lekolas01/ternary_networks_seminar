import unittest
from bool_formula import Constant, Literal, NOT, AND, OR, all_interpretations, overlap
import torch.nn as nn
from nn_to_bool_formula import full_circle


class TestFullCircle(unittest.TestCase):
    def test_binaryFunctions(self):
        target_funcs = [
            Constant(False),
            Constant(True),
            NOT(Literal("x1")),
            Literal("x1"),
            AND(NOT("x1"), NOT("x2")),
            AND(NOT("x1"), "x2"),
            AND("x1", NOT("x2")),
            AND("x1", "x2"),
            OR(NOT("x1"), NOT("x2")),
            OR(NOT("x1"), "x2"),
            OR("x1", NOT("x2")),
            OR("x1", "x2"),
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
                AND("x1", NOT("x2")),
            ),
            OR(
                AND(NOT("x1"), NOT("x2")),
                AND("x1", "x2"),
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
            found_func = ans["bool_graph"]
            names = target_func.all_literals().union(found_func.all_literals())
            data = all_interpretations(names)
            assert (
                overlap(target_func, found_func, data) >= 0.8
            ), f"Did not produce an equivalent function: {target_func = }; {found_func = }"
            break
