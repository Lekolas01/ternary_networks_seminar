import numpy as np

from neuron import IfThenRule, Subproblem

# rule = IfThenRule("y", [("x1", True), ("x2", False)])
# knowledge = {"x2": False}

# print(rule)
# print(rule.simplify(knowledge))
# print(knowledge)
# print(rule)
#
# print(rule({"x1": np.array([False, False]), "x2": np.array([True, True])}))

rule1 = IfThenRule("y", [("y1", True)])
rule2 = IfThenRule("y", [("x1", True), ("y2", True)])
sp = Subproblem("y", [rule1, rule2])
print(sp)
# this makes the first rule T, so the whole subproblem should be T, as rule1 will always trigger
knowledge = {"y1": True}
print(sp.simplify(knowledge))
print(f"{sp = }")
print(f"{knowledge = }")
