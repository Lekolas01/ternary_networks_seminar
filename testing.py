from bool_formula import Example
from neuron import *

q_ng = QuantizedNeuronGraph(
    [
        QuantizedNeuron(
            "y",
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

bg = RuleSetGraph.from_q_neuron_graph(q_ng)
print(bg)

a = OR(Constant(np.array(False)), Constant(np.array(False)), Literal("x1"))
print(a)
print(a.simplified())
y = AND("x1", "x2")

print(y)
