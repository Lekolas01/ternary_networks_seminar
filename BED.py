import functools
from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableMapping
from graphlib import TopologicalSorter
from typing import Self

import numpy as np

from bool_formula import possible_data
from utilities import invert_dict

Interpretation = MutableMapping[str, np.ndarray]


class Vertex(ABC):
    def __init__(self, key: str, parents: set[Self], children: set[Self]):
        self.key = key
        self.parents = set()
        for p in parents:
            connect(p, self)

        self.children = set()
        for c in children:
            connect(self, c)

    @abstractmethod
    def __call__(self, vars: Interpretation) -> np.ndarray:
        pass

    def simplified(self) -> Self:
        return self

    def __str__(self) -> str:
        return self.key

    def __repr__(self) -> str:
        return self.key


def connect(parent_node: Vertex, child_node: Vertex):
    parent_node.children.add(child_node)
    child_node.parents.add(parent_node)


def disconnect(parent_node: Vertex, child_node: Vertex):
    parent_node.children.remove(child_node)
    child_node.parents.remove(parent_node)


class Terminal(Vertex):
    def __init__(self, val: bool, parents: set[Self] = set()):
        super().__init__("T", parents, set())
        self.val = val

    def __call__(self, vars: Interpretation) -> np.ndarray:
        return np.array(self.val)


class Variable(Vertex):
    def __init__(self, key: str, parents: set[Self] = set()):
        super().__init__(key, parents, set())

    def __call__(self, vars: Interpretation) -> np.ndarray:
        return vars[self.key]


class Operator(Vertex):
    def __init__(
        self,
        key: str,
        is_all: bool,
        parents: set[Self] = set(),
        children: set[Self] = set(),
    ) -> None:
        super().__init__(key, parents, children)
        self.key = key
        self.is_all = is_all
        self.op = np.all if self.is_all else np.any
        self.opstr = " & " if is_all else " | "

    def __call__(self, vars: Interpretation) -> np.ndarray:
        answers = [c(vars) for c in self.children]
        if self.is_all:
            return functools.reduce(lambda x, y: x & y, answers)
        return functools.reduce(lambda x, y: x | y, answers)

    def __str__(self) -> str:
        if len(self.children) == 0:
            return "(T)" if self.is_all else "(F)"
        else:
            return f"({self.opstr.join([c.__str__() for c in self.children])})"

    def simplified(self) -> Vertex:
        # simplify children first
        children = set([c.simplified() for c in self.children])
        # if one term is np.array(False), the whole conjunction is np.array(False)
        if any(child == (not self.is_all) for child in children):
            return Terminal(not self.is_all)
        # True constants can be removed
        children = set(filter(lambda c: c != self.is_all, children))
        # if now the list is empty, we can return True
        if len(children) == 0:
            return Terminal(self.is_all)
        if len(children) == 1:
            return list(children)[0]

        new_children = set()
        return Operator(self.key, self.is_all, self.parents, children)
        for child in children:
            # if a child is the same quantifier type, you can combine it with the parent
            # for example: AND(AND(a, b), c) -> AND(a, b, c)
            if isinstance(child, Operator) and child.is_all == self.is_all:
                for c in child.children:
                    new_children.append(c)
            else:
                new_children.append(child)

        # otherwise return the rest of the relevant children
        return Operator(new_children, self.is_all)


class BED:
    def __init__(self, vertices: list[Vertex], root: Vertex):
        self.vertices = vertices
        self.v_children = {v: v.children for v in vertices}
        self.v_parents = invert_dict(self.v_children)
        assert root in vertices
        self.root = root

    def topological_order(self) -> list[Vertex]:
        sorter = TopologicalSorter(self.v_children)
        return list(sorter.static_order())

    def __call__(self, vars: Interpretation) -> np.ndarray:
        for vertex in self.topological_order():
            vars[vertex.key] = vertex(vars)
        return vars[self.root.key]


a = Terminal(True)
# a = Variable("x3", [])
b = Variable("x2", set())
# c = Quantifier("y", True, set(), {a, b})

y = Operator("y", False)
y1 = Variable("y1")
y_T = Operator("y_T", True)
x1 = Variable("x1")
y2 = Variable("y2")
print(y)
connect(y, y1)
connect(y, y_T)
connect(y_T, x1)
connect(y_T, y2)

d = BED([y, y1, y_T, x1, y2], y)
target = Operator("target", False)
target2 = Operator("target2", False)
a = Variable("a")
b = Variable("b")
c = Variable("c")
bed = BED([a, b, c, target, target2], target)

keys = ["a", "b", "c"]
data = possible_data(keys, is_float=False)

print(bed(data))
