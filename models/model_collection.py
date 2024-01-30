from dataclasses import dataclass

import torch.nn as nn

from neuron import Activation


@dataclass
class LayerSpec:
    shape: int
    act_fn: Activation


NNSpec = list[tuple[int, Activation]]


class ModelFactory:
    specifications: dict[str, NNSpec] = {
        "lin_db": [(1, Activation.SIGMOID)],
        "abcdefg": [(3, Activation.TANH), (1, Activation.SIGMOID)],
        "parity2": [(2, Activation.SIGMOID), (1, Activation.SIGMOID)],
        "parity9": [(9, Activation.SIGMOID), (1, Activation.SIGMOID)],
        "parity10": [(10, Activation.SIGMOID), (1, Activation.SIGMOID)],
        "deep_parity10": [
            (5, Activation.TANH),
            (5, Activation.TANH),
            (1, Activation.SIGMOID),
        ],
    }

    @classmethod
    def get_model(cls, spec_name: str, in_shape: int) -> nn.Sequential:
        nn_spec = cls.specifications[spec_name]
        ans = nn.Sequential()
        last_shape = in_shape
        for idx, layer_spec in enumerate(nn_spec):
            ans.add_module(f"lin{idx}", nn.Linear(last_shape, layer_spec[0]))
            act_fn = nn.Tanh() if layer_spec[1] == Activation.TANH else nn.Sigmoid()
            ans.add_module(f"act_fn{idx}", act_fn)
        ans.add_module(f"flatten", nn.Flatten(0))
        return ans
