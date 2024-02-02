import numpy as np

from neuron import IfThenRule, Subproblem

print(range(-2))
for val in range(-5):
    print(val)

rule = IfThenRule("y", [("x1", True), ("x2", False)])

print(rule)
print(rule.simplify({"x2": False}))
print(rule)
print(rule.simplify({"x2": False}))
print(rule)

print(rule({"x1": np.array([True, False]), "x2": np.array([True, True])}))

rule1 = IfThenRule("y", [("y1", True)])
rule2 = IfThenRule("y", [("x1", True), ("y2", True)])
sp = Subproblem("y", [rule1, rule2])
print(sp)
print(sp.simplify({}))
