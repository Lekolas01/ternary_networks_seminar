from __future__ import annotations
from typing import Dict
import random
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
    def negate(self) -> None:
        raise NotImplementedError

    def simplify(self) -> None:
        pass

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

    def negate(self) -> None:
        self.val = not self.val


class Literal(Bool):
    def __init__(self, name: str, is_true: bool = True) -> None:
        assert name is not None
        self.name = name
        self.value = is_true

    def __call__(self, interpretation: Interpretation) -> bool:
        ans = interpretation[self.name]
        return ans if self.value else not ans

    def __str__(self) -> str:
        return f"{'!' if not self.value else ''}{self.name}"

    def all_literals(self) -> set[str]:
        return {self.name}

    def negate(self) -> None:
        self.value = not self.value


class NOT(Bool):
    def __init__(self, child: Bool) -> None:
        self.child = child

    def __call__(self, interpretation: Interpretation) -> bool:
        return not self.child(interpretation)

    def __str__(self) -> str:
        return f"NOT({self.child.__str__()})"

    def all_literals(self) -> set[str]:
        return self.child.all_literals()

    def negate(self) -> None:
        self = self.child

    def simplify(self) -> None:
        self.child.negate()
        self = self.child


class Quantifier(Bool):
    def __init__(self, children: list[Bool], is_all: bool) -> None:
        self.children = children
        self.set_is_all(is_all)

    def __call__(self, interpretation: Interpretation) -> bool:
        return self._op(c(interpretation) for c in self.children)

    def __str__(self) -> str:
        return f"{self._oprepr}({', '.join([c.__str__() for c in self.children])})"

    def all_literals(self) -> set[str]:
        return set().union(*(f.all_literals() for f in self.children))

    def get_is_all(self) -> bool:
        return self._is_all

    def set_is_all(self, is_all: bool) -> None:
        if hasattr(self, "is_all") and self._is_all == is_all:
            return
        self._is_all = is_all
        self._opstr = " & " if is_all else " | "
        self._oprepr = "AND" if is_all else "OR"
        self._op = all if self._is_all else any


class AND(Quantifier):
    def __init__(self, *children: Bool) -> None:
        super().__init__(list(children), True)

    def simplify(self) -> None:
        # TODO: simplify children first
        # if one term is False, the whole conjunction is False
        if any(child == False for child in self.children):
            self = Constant(False)
            return
        # True constants can be removed
        children = list(filter(lambda c: c != True, self.children))
        # if now the list is empty, we can return True
        if len(children) == 0:
            self = Constant(True)
            return
        if len(children) == 1:
            self = children[0]
            return
        # otherwise return the rest of the relevant children
        self = AND(*children)
        return

    def negate(self) -> None:
        for c in self.children:
            c.negate()
        self = OR(*self.children)


class OR(Quantifier):
    def __init__(self, *children: Bool) -> None:
        super().__init__(list(children), False)

    def simplify(self) -> Bool:
        # if one term is True, the whole disjuction is True
        if any(child == True for child in self.children):
            return Constant(True)
        # False constants can be removed
        children = list(filter(lambda c: c != False, self.children))
        # if now the list is empty, we can return False
        if len(self.children) == 0:
            return Constant(False)
        if len(children) == 1:
            return children[0]
        # otherwise return the rest of the relevant children
        return OR(*children)

    def negate(self) -> None:
        for c in self.children:
            c.negate()
        self = OR(*self.children)


if __name__ == "__main__":
    interpretation = {"a": True, "b": False, "c": False}
    z = Constant(True)
    a = Literal("a")
    b = Literal("b")
    c = Literal("c")
    d = OR(a, b)
    func = OR(OR(a, b), c)
    fn1 = AND(a, b)
    print(func(interpretation))
    print(func)
    print(func.simplify())
