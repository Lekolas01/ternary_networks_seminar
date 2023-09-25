from __future__ import annotations
from typing import Dict
from abc import ABC, abstractmethod
from collections.abc import Collection
import copy

Interpretation = Dict[str, bool]


def all_interpretations(names: Collection[str]) -> list[Interpretation]:
    ans = []
    for i in range(2 ** len(names)):
        interpretation = {name: ((i >> idx) % 2 == 1) for idx, name in enumerate(names)}
        ans.append(interpretation)
    return ans


def fidelity(left: Bool, right: Bool, verbose=False) -> float:
    names = list(left.all_literals().union(right.all_literals()))
    interpretations = all_interpretations(names)
    if verbose:
        print(f"{names = }")
        print(f"{len(interpretations) = }")

    ans = 0
    for inter in interpretations:
        if left(inter) == right(inter):
            ans += 1
    return ans / len(interpretations)


class Bool(ABC):
    @abstractmethod
    def __call__(self, interpretation: Interpretation) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def all_literals(self) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def negated(self) -> Bool:
        return NOT(copy.copy(self))

    def simplified(self) -> Bool:
        return self

    def __eq__(self, other: Bool | bool) -> bool:
        if isinstance(other, bool):
            return fidelity(self, Constant(other)) == 1
        return fidelity(self, other) == 1


class Constant(Bool):
    def __init__(self, is_true: bool) -> None:
        self.val = is_true

    def __call__(self, interpretation: Interpretation) -> bool:
        return self.val

    def __str__(self) -> str:
        return "T" if self.val else "F"

    def all_literals(self) -> set[str]:
        return set()

    def negated(self) -> Bool:
        return Constant(not self.val)


class Literal(Bool):
    def __init__(self, name: str) -> None:
        assert name is not None
        self.name = name

    def __call__(self, interpretation: Interpretation) -> bool:
        ans = interpretation[self.name]
        return ans

    def __str__(self) -> str:
        return self.name

    def all_literals(self) -> set[str]:
        return {self.name}

    def negated(self) -> Bool:
        return NOT(copy.copy(self))


class NOT(Bool):
    def __init__(self, child: Bool) -> None:
        self.child = child

    def __call__(self, interpretation: Interpretation) -> bool:
        return not self.child(interpretation)

    def __str__(self) -> str:
        return f"!{self.child.__str__()}"

    def all_literals(self) -> set[str]:
        return self.child.all_literals()

    def negated(self) -> Bool:
        # if you negate a NOT(...), you can just cancel the NOT() and return the child
        return copy.copy(self.child)

    def simplified(self) -> Bool:
        # move the NOT() inside
        if isinstance(self.child, Literal):
            return copy.copy(self)
        return self.child.negated().simplified()


class Quantifier(Bool):
    def __init__(self, children: Collection[Bool | str], is_all: bool) -> None:
        self.children = [c if isinstance(c, Bool) else Literal(c) for c in children]
        self.is_all = is_all
        self.op = all if self.is_all else any
        self.opstr = " & " if is_all else " | "

    def __call__(self, interpretation: Interpretation) -> bool:
        return self.op(c(interpretation) for c in self.children)

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

    def simplified(self) -> Bool:
        # simplify children first
        children = [c.simplified() for c in self.children]
        # if one term is False, the whole conjunction is False
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
    def __init__(self, *children: Bool | str) -> None:
        super().__init__(list(children), True)


class OR(Quantifier):
    def __init__(self, *children: Bool | str) -> None:
        super().__init__(list(children), False)


class PARITY(Bool):
    def __init__(self, literals: Collection[str]) -> None:
        super().__init__()
        self.literals = set(literals)

    def __call__(self, interpretation: Interpretation) -> bool:
        assert all(literal in interpretation for literal in self.literals)
        return sum(1 for key in interpretation if interpretation[key]) % 2 == 1

    def __str__(self) -> str:
        return f"PARITY({','.join([c.__str__() for c in self.literals])})"

    def all_literals(self) -> set[str]:
        return self.literals

    def negated(self) -> Bool:
        return super().negated()


if __name__ == "__main__":
    formulae: Dict[str, Bool] = {
        "a0": AND(),
        "a1": AND(Constant(True)),
        "a2": AND(Constant(False)),
        "a": AND(Constant(True), Constant(True)),
        "b": AND(Constant(True), Constant(False)),
        "c": AND(Constant(False), Constant(False)),
        "d": AND(Constant(True), Constant(True), Literal("x1")),
        "e": AND(Constant(True), Constant(False), Literal("x1")),
        "f": AND(Constant(False), Constant(False), Literal("x1")),
        "n1": NOT(Constant(True)),
        "n2": NOT(NOT(Constant(True))),
        "n3": NOT(Constant(False)),
        "n4": NOT(NOT(Constant(False))),
        "n5": NOT(Literal("x1")),
        "n6": NOT(NOT(Literal("x1"))),
        "or0": OR(),
        "and1": NOT(AND()),
        "and2": NOT(AND(Literal("x1"))),
        "and3": NOT(AND(Literal("x1"), Literal("x2"))),
        "and4": AND(AND(Constant(True), Literal("x1")), Constant(True)),
        "and5": AND("x1", "x2", AND("x3", "x4", AND("x5"))),
    }
    for key in formulae:
        b = formulae[key]
        ans = b.simplified()
        print(f"{key:>6} | {b.__str__():>15} -> {ans.__str__():>5}")
