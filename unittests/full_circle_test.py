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
                nn.Linear(4, 1),
                nn.Sigmoid(),
                nn.Flatten(0),
            )
            found = full_circle(target_func, model, epochs=3)
            assert (
                target_func == found
            ), f"Did not produce an equivalent function: {target_func = }; {found = }"

    def test_XOR(self):
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
            found_func = full_circle(target_func, model, epochs=30)
            assert (
                target_func == found_func
            ), f"Did not produce an equivalent function: {target_func = }; {found_func = }"
