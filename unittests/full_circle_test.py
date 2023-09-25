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
                AND(NOT(Literal("x1")), Literal("x2")),
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
            found_func = full_circle(target_func, model, epochs=30)
            assert (
                target_func == found_func
            ), f"Did not produce an equivalent function: {target_func = }; {found_func = }"
            break

    def test_parity(self):
        parity = PARITY(("x1", "x2", "x3", "x4"))
        model = nn.Sequential(
            nn.Linear(6, 6),
            nn.Sigmoid(),
            nn.Linear(6, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid(),
            nn.Flatten(0),
        )
        found = full_circle(parity, model, epochs=1, verbose=True)
        temp = 0
