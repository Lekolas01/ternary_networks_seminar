from bool_formula import AND, OR, Constant, Example, Literal, possible_data
from neuron import *

q_ng = QuantizedNeuronGraph(
    [
        QuantizedNeuron(
            "target",
            {
                "x1": 5.0,
                "x2": 5.0,
                "x3": 1.0,
                "x4": 1.0,
                "x5": 1.0,
                "x6": 1.0,
                "x7": 1.0,
            },
            -9.5,
        )
    ]
)


bg = RuleSetGraph.from_q_neuron_graph(q_ng, simplify=False)
print(bg)
simple_bg = RuleSetGraph.from_q_neuron_graph(q_ng, simplify=True)
print(simple_bg)
data = possible_data([f"x{i + 1}" for i in range(7)], is_float=False)
print(bg(data) == simple_bg(data))
