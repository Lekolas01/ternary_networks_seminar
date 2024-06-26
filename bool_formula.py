from __future__ import annotations

import copy
import functools
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Mapping, MutableMapping
from enum import Enum
from functools import reduce
from graphlib import TopologicalSorter
from typing import Callable, Dict

import numpy as np


class Activation(Enum):
    SIGMOID = 1
    TANH = 2


Interpretation = MutableMapping[str, np.ndarray]


def overlap(left: Callable, right: Callable, data: Collection) -> float:
    ans = 0
    for datapoint in data:
        if left(datapoint) == right(datapoint):
            ans += 1
    return ans / len(data)


def possible_data(
    keys: Collection[str], is_float=True, shuffle=False
) -> Dict[str, np.ndarray]:
    ans: dict[str, np.ndarray] = {}
    n = len(keys)
    for i, key in enumerate(keys):
        a = (
            np.concatenate((np.repeat(0.0, 2**i), np.repeat(1.0, 2**i)))
            if is_float
            else np.concatenate((np.repeat(False, 2**i), np.repeat(True, 2**i)))
        )
        ans[key] = np.tile(a, 2 ** (n - i - 1))
    if shuffle:
        order = np.random.permutation(2**n)
        for i, key in enumerate(keys):
            ans[key] = ans[key][order]
    return ans


def all_data(keys: Collection[str], low, high, shuffle=False) -> Dict[str, np.ndarray]:
    ans: dict[str, np.ndarray] = {}
    n = len(keys)
    for i, key in enumerate(keys):
        a = np.concatenate((np.repeat(low, 2**i), np.repeat(high, 2**i)))
        ans[key] = np.tile(a, 2 ** (n - i - 1))
    if shuffle:
        order = np.random.permutation(2**n)
        for i, key in enumerate(keys):
            ans[key] = ans[key][order]
    return ans


class Bool(ABC):
    @abstractmethod
    def __call__(self, interpretation: Mapping[str, np.ndarray] = {}) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def all_literals(self) -> set[str]:
        raise NotImplementedError

    def negated(self) -> Bool:
        return NOT(copy.copy(self))

    def simplified(self, knowledge: Knowledge = {}) -> Bool:
        return self

    def __eq__(self, other: Bool | bool) -> bool:
        names = (
            self.all_literals().union(other.all_literals())
            if isinstance(other, Bool)
            else self.all_literals()
        )
        data = possible_data(names, is_float=False)
        return bool(np.all(self(data)))

    def __repr__(self) -> str:
        return str(self)


Knowledge = dict[str, Bool]


class Constant(Bool):
    def __init__(self, val: np.ndarray | bool) -> None:
        if isinstance(val, bool):
            val = np.array(val, dtype=bool)
        self.val = val

    def __call__(self, interpretation: Mapping[str, np.ndarray] = {}) -> np.ndarray:
        return self.val

    def __str__(self) -> str:
        return "T" if self.val else "F"

    def all_literals(self) -> set[str]:
        return set()

    def negated(self) -> Bool:
        return Constant(~self.val)

    def __eq__(self, other: Bool | bool) -> bool:
        if isinstance(other, bool):
            return self.val == other
        if isinstance(other, Constant):
            return self.val == other.val
        return super().__eq__(other)


class Literal(Bool):
    def __init__(self, name: str) -> None:
        assert name is not None
        self.name = name

    def __call__(self, interpretation: Mapping[str, np.ndarray] = {}) -> np.ndarray:
        ans = interpretation[self.name]
        return ans

    def __str__(self) -> str:
        return self.name

    def all_literals(self) -> set[str]:
        return {self.name}

    def negated(self) -> Bool:
        return NOT(copy.copy(self))


def val_2_Bool(val: Bool | str | bool) -> Bool:
    if isinstance(val, Bool):
        return val
    if isinstance(val, str):
        return Literal(val)
    return Constant(val)


class NOT(Bool):
    def __init__(self, child: Bool | str | bool) -> None:
        self.child = val_2_Bool(child)

    def __call__(self, interpretation: Mapping[str, np.ndarray] = {}) -> np.ndarray:
        ans = self.child(interpretation)
        return ~(ans)

    def __str__(self) -> str:
        return f"!{self.child.__str__()}"

    def all_literals(self) -> set[str]:
        return self.child.all_literals()

    def negated(self) -> Bool:
        return copy.copy(self.child)

    def simplified(self, knowledge={}) -> Bool:
        c = self.child.simplified()
        if not isinstance(c, Quantifier):
            return c.negated()
        return c.negated()


class Quantifier(Bool):
    def __init__(self, children: Collection[Bool | str | bool], is_all: bool) -> None:
        self.children = [val_2_Bool(c) for c in children]
        self.is_all = is_all
        self.op = np.all if self.is_all else np.any
        self.opstr = " & " if is_all else " | "

    def __call__(self, interpretation: Mapping[str, np.ndarray] = {}) -> np.ndarray:
        answers = [c(interpretation) for c in self.children]
        return functools.reduce(lambda x, y: (x & y if self.is_all else x | y), answers)

    def __str__(self) -> str:
        if len(self.children) == 0:
            return "(T)" if self.is_all else "(F)"
        else:
            return f"({self.opstr.join([c.__str__() for c in self.children])})"

    def all_literals(self) -> set[str]:
        return set().union(*(f.all_literals() for f in self.children))

    def negated(self) -> Bool:
        return Quantifier(
            [c.negated().simplified() for c in self.children], not self.is_all
        )

    def simplified(self, knowledge: Knowledge = {}) -> Bool:
        # simplify children first
        children = [c.simplified(knowledge) for c in self.children]
        # if one term is np.array(False), the whole conjunction is np.array(False)
        if any(child == (not self.is_all) for child in children):
            return Constant(not self.is_all)
        # True constants can be removed
        children = list(filter(lambda c: c != self.is_all, children))
        # if now the list is empty, we can return True
        if len(children) == 0:
            return Constant(self.is_all)
        if len(children) == 1:
            return children[0]

        new_children = []
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


class AND(Quantifier):
    def __init__(self, *children: Bool | str | bool) -> None:
        super().__init__(list(children), True)


class OR(Quantifier):
    def __init__(self, *children: Bool | str | bool) -> None:
        super().__init__(list(children), False)


class PARITY(Bool):
    def __init__(self, literals: Iterable[str]) -> None:
        super().__init__()
        self.literals = set(literals)

    def __call__(self, interpretation: Mapping[str, np.ndarray] = {}) -> np.ndarray:
        return reduce(lambda x, y: x ^ y, (interpretation[l] for l in self.literals))

    def __str__(self) -> str:
        return f"PARITY({','.join([c.__str__() for c in self.literals])})"

    def all_literals(self) -> set[str]:
        return self.literals


class Example(Bool):
    def __init__(self) -> None:
        super().__init__()
        self.literals = {"x1", "x2", "x3", "x4", "x5"}

    def __call__(self, vars: Mapping[str, np.ndarray] = {}) -> np.ndarray:
        t1 = vars["x3"] & vars["x4"] & vars["x5"]
        return (~vars["x1"] & vars["x2"]) | (~vars["x1"] & t1) | (vars["x2"] & t1)

    def __str__(self) -> str:
        return ""

    def all_literals(self) -> set[str]:
        return super().all_literals()


class BED:
    def __init__(self, nodes: dict[str, Bool]):
        self.nodes = nodes

    def topological_order(self) -> list[str]:
        graph_ins = {key: node.all_literals() for key, node in self.nodes.items()}
        # TODO: remove constants and literals??
        sorter = TopologicalSorter(graph_ins)
        return list(sorter.static_order())

    def __call__(self, vars: Interpretation) -> np.ndarray:
        for key in self.topological_order():
            node = self.nodes[key]
            vars[key] = node(vars)
        return vars[self.target_key()]

    def target_key(self):
        return self.topological_order()[-1]

    def __str__(self) -> str:
        return "\n".join(f"{key} := {str(node)}" for key, node in self.nodes.items())

    def simplify(self) -> None:
        knowledge = {}
        easy_root = self.nodes[self.target_key()].simplified(knowledge)
        print(knowledge)
        print(easy_root)


if __name__ == "__main__":
    formulae: Dict[str, Bool] = {
        "a0": AND(),
        "a1": AND(Constant(np.array(True))),
        "a2": AND(Constant(np.array(False))),
        "a": AND(Constant(np.array(True)), Constant(np.array(True))),
        "b": AND(Constant(np.array(True)), Constant(np.array(False))),
        "c": AND(Constant(np.array(False)), Constant(np.array(False))),
        "d": AND(Constant(np.array(True)), Constant(np.array(True)), Literal("x1")),
        "e": AND(Constant(np.array(True)), Constant(np.array(False)), Literal("x1")),
        "f": AND(Constant(np.array(False)), Constant(np.array(False)), Literal("x1")),
        "n1": NOT(Constant(np.array(True))),
        "n2": NOT(NOT(Constant(np.array(True)))),
        "n3": NOT(Constant(np.array(False))),
        "n4": NOT(NOT(Constant(np.array(False)))),
        "n5": NOT(Literal("x1")),
        "n6": NOT(NOT(Literal("x1"))),
        "or0": OR(),
        "and1": NOT(AND()),
        "and2": NOT(AND(Literal("x1"))),
        "and3": NOT(AND(Literal("x1"), Literal("x2"))),
        "and4": AND(
            AND(Constant(np.array(True)), Literal("x1")), Constant(np.array(True))
        ),
        "and5": AND("x1", "x2", AND("x3", "x4", AND("x5"))),
    }
    for key in formulae:
        b = formulae[key]
        ans = b.simplified()
        # print(f"{key:>6} | {b.__str__():>15} -> {ans.__str__():>5}")

    y = NOT(True)
    print(y)
    print(y.simplified(), "\n")

    y = NOT("x1")
    print(y)
    print(y.simplified(), "\n")

    y = NOT(NOT("x1"))
    print(y)
    print(y.simplified(), "\n")
