from neuron import BooleanGraph, QuantizedNeuron, QuantizedNeuronGraph

a = [3, 1, 5, 2]


keys = [f"x{i + 1}" for i in range(5)]

h1 = QuantizedNeuron(
    "h1",
    {"x1": 3.0, "x2": 3.0, "x3": 1.0, "x4": 1.0, "x5": 1.0},
    -5.5,
)
q_ng = QuantizedNeuronGraph([h1])
b_ng = BooleanGraph.from_q_neuron_graph(q_ng)
print(b_ng)
