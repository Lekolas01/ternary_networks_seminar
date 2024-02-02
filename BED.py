from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Self

import numpy as np

Interpretation = MutableMapping[str, np.ndarray]


class Node(ABC):
    def __init__(self, parents: list[Self], children: list[Self]):
        self.parents = parents
        self.children = children

    @abstractmethod
    def __call__(self, vars: Interpretation) -> np.ndarray:
        pass

    def simplified(self) -> Self:
        return self


class Const(Node):
    def __init__(self, val: bool):
        super().__init__([], [])
        self.val = val

    def __call__(self, vars: Interpretation) -> np.ndarray:
        return np.ndarray(self.val)


class Variable(Node):
    def __init__(self, key: str, parents: list[Node]):
        super().__init__(parents, [])
        self.key = key

    def __call__(self, vars: Interpretation) -> np.ndarray:
        return vars[self.key]


class Operator(Node):
    def __init__(
        self, key: str, is_all: bool, parents: list[Node], children: list[Node]
    ) -> None:
        # super(self).__init__(parents, children)
        self.parents = parents
        self.children = children
        self.key = key
        self.is_all = is_all
        self.op = np.all if self.is_all else np.any
        self.opstr = " & " if is_all else " | "

    def __call__(self, vars: Interpretation) -> np.ndarray:
        answers = np.stack([c(vars) for c in self.children], axis=1)
        return self.op(answers, axis=1)

    def __str__(self) -> str:
        if len(self.children) == 0:
            return "(T)" if self.is_all else "(F)"
        else:
            return f"({self.opstr.join([c.__str__() for c in self.children])})"

    def simplified(self) -> Node:
        # simplify children first
        children = [c.simplified() for c in self.children]
        # if one term is np.array(False), the whole conjunction is np.array(False)
        if any(child == (not self.is_all) for child in children):
            return Const(not self.is_all)
        # True constants can be removed
        children = list(filter(lambda c: c != self.is_all, children))
        # if now the list is empty, we can return True
        if len(children) == 0:
            return Const(self.is_all)
        if len(children) == 1:
            return children[0]

        new_children = []
        return Operator(self.key, self.is_all, self.parents, children)
        for child in children:
            # if a child is the same quantifier type, you can combine it with the parent
            # for example: AND(AND(a, b), c) -> AND(a, b, c)
            if isinstance(child, Quantifier) and child.is_all == self.is_all:
                for c in child.children:
                    new_children.append(c)
            else:
                new_children.append(child)

        # otherwise return the rest of the relevant children
        return Quantifier(new_children, self.is_all)
