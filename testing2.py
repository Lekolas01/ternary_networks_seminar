import graphviz
import graphviz.dot as d
from pyeda.boolalg.bdd import _NODES
from pyeda.inter import *

print(len(_NODES))
x1, x2, x3, x4, x5 = map(bddvar, (f"x{i + 1}" for i in range(5)))
print(len(_NODES))
f = expr("x1 & x2 | x1 & x3 & x4 & x5 | x2 & x3 & x4 & x5")
print(f)
f = expr2bdd(f)
print(f)
dot = f.to_dot()

with open("Output.txt", "w") as f:
    f.write(dot)
