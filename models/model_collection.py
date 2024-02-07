from dataclasses import dataclass

import torch.nn as nn

from neuron import Activation

NNSpec = list[tuple[int, int, Activation]]


class ModelFactory:
    specifications: dict[str, NNSpec] = {
        "lin_db": [(1, 1, Activation.SIGMOID)],
        "abcdefg": [(7, 3, Activation.TANH), (3, 1, Activation.SIGMOID)],
        "abcdefg_2_dead": [(9, 3, Activation.TANH), (3, 1, Activation.SIGMOID)],
        "abcdefg_small": [(7, 2, Activation.TANH), (2, 1, Activation.SIGMOID)],
        "parity2": [(2, 2, Activation.SIGMOID), (2, 1, Activation.SIGMOID)],
        "parity5": [(5, 5, Activation.SIGMOID), (5, 1, Activation.SIGMOID)],
        "parity9": [(9, 9, Activation.SIGMOID), (9, 1, Activation.SIGMOID)],
        "parity10": [(10, 10, Activation.TANH), (10, 1, Activation.SIGMOID)],
        "deep_parity10": [
            (10, 6, Activation.TANH),
            (6, 6, Activation.TANH),
            (6, 1, Activation.SIGMOID),
        ],
    }

    @classmethod
    def get_model(cls, spec_name: str) -> nn.Sequential:
        nn_spec = cls.specifications[spec_name]
        ans = nn.Sequential()
        for idx, layer_spec in enumerate(nn_spec):
            ans.add_module(f"lin{idx}", nn.Linear(layer_spec[0], layer_spec[1]))
            act_fn = nn.Tanh() if layer_spec[2] == Activation.TANH else nn.Sigmoid()
            ans.add_module(f"act_fn{idx}", act_fn)
        ans.add_module(f"flatten", nn.Flatten(0))
        return ans
