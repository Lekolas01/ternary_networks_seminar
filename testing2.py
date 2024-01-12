import bisect

from neuron import QuantizedNeuron, RuleSetNeuron

a = [(str(i), 20 - i + (4 if i % 2 == 0 else 0)) for i in range(1, 10)]
print(a)
a = sorted(a, key=lambda x: x[1])
print(a)

bisect.insort(a, ("12", 12), key=lambda x: x[1])
print(a)
