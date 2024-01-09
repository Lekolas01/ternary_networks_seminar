from neuron import BooleanNeuron, QuantizedNeuron

a = [3, 1, 5, 2]


keys = [f"x{i + 1}" for i in range(5)]

h1 = QuantizedNeuron(
    "h1",
    {"x1": 3.0, "x2": 3.0, "x3": 1.0, "x4": 1.0, "x5": 1.0},
    -5.2,
)
bn = BooleanNeuron(h1)

print(h1)
print(bn)
