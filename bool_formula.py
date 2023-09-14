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

    def __eq__(self, other: Bool) -> bool:
        if isinstance(other, Bool):
            names = list(self.all_literals().union(other.all_literals()))
            for interpretation in all_interpretations(names):
                if self(interpretation) != other(interpretation):
                    return False
            return True
        elif isinstance(other, bool):
            names = list(self.all_literals())
            for interpretation in all_interpretations(names):
                if self(interpretation) != other:
                    return False
            return True


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
    def __init__(self, children: list[Bool], is_all: bool) -> None:
        self.children = children
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
        # otherwise return the rest of the relevant children
        return Quantifier(children, self.is_all)


class AND(Quantifier):
    def __init__(self, *children: Bool) -> None:
        super().__init__(list(children), True)


class OR(Quantifier):
    def __init__(self, *children: Bool) -> None:
        super().__init__(list(children), False)


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
    }
    for key in formulae:
        b = formulae[key]
        ans = b.simplified()
        print(f"{key:>6} | {b.__str__():>15} -> {ans.__str__():>5}")
