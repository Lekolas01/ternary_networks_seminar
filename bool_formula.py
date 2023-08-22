from __future__ import annotations
from typing import Any, Dict
import random


Interpretation = Dict[str, bool]


class Boolean:
    def __init__(self) -> None:
        pass

    def __call__(self, interpretation: Interpretation) -> bool:
        return False

    def __eq__(self, other: Boolean) -> bool:
        literals = list(self.all_literals().union(other.all_literals()))
        n_literals = len(literals)

        for i in range(2**n_literals):
            interpretation = dict(
                [(l, ((i >> idx) % 2 == 1)) for idx, l in enumerate(literals)]
            )
            if self(interpretation) != other(interpretation):
                return False
        return True

    def __repr__(self) -> str:
        return str(self)

    def all_literals(self) -> set[str]:
        return set()


class Constant(Boolean):
    def __init__(self, val: bool) -> None:
        self.val = val

    def __str__(self) -> str:
        return "T" if self.val else "F"

    def __call__(self, interpretation: Interpretation) -> bool:
        return self.val

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Constant) and __value.val == self.val

    def all_literals(self) -> set[str]:
        return set()


class Literal(Boolean):
    def __init__(self, value: str, positive=True) -> None:
        assert value is not None
        self.name = value
        self.positive = positive

    def __str__(self) -> str:
        return f"{'!' if not self.positive else ''}{str(self.name)}"

    def __call__(self, interpretation: Interpretation) -> bool:
        ans = interpretation[self.name]
        if not self.positive:
            ans = not ans
        return ans

    def all_literals(self) -> set[str]:
        return {self.name}


class Quantifier(Boolean):
    def __init__(self, children: list[Boolean], op_str: str) -> None:
        super().__init__()
        self.children = children
        self.op_str = op_str

    def __str__(self) -> str:
        return f"({f' {self.op_str} '.join([str(c) for c in self.children])})"

    def __call__(self, interpretation: Interpretation) -> bool:
        return super().__call__(interpretation)

    def all_literals(self) -> set[str]:
        return set().union(*(f.all_literals() for f in self.children))


class AND(Quantifier):
    def __init__(self, children: list[Boolean]) -> None:
        super().__init__(children, "&")

    def __call__(self, interpretation: Interpretation) -> bool:
        temp = [c(interpretation) for c in self.children]
        return all(temp)


class OR(Quantifier):
    def __init__(self, children: list[Boolean]) -> None:
        super().__init__(children, "|")

    def __call__(self, interpretation: Interpretation) -> bool:
        temp = [c(interpretation) for c in self.children]
        return any(temp)


def simplified(b: Boolean) -> Boolean:
    if isinstance(b, Quantifier):
        for i, child in enumerate(b.children):
            b.children[i] = simplified(child)
        if isinstance(b, AND):
            # if one term is False, the whole conjunction is False
            if any(child == Constant(False) for child in b.children):
                return Constant(False)

            # True constants can be removed
            b.children = list(filter(lambda c: c != Constant(True), b.children))
            # if now the list is empty, we can return True
            if len(b.children) == 0:
                return Constant(True)
            # otherwise return b
        if isinstance(b, OR):
            # if one term is True, the whole conjunction is True
            if any(child == Constant(True) for child in b.children):
                return Constant(True)

            # False constants can be removed
            b.children = list(filter(lambda c: c != Constant(False), b.children))
            # if now the list is empty, we can return False
            if len(b.children) == 0:
                return Constant(False)
            # otherwise return b
        if len(b.children) == 1:
            return b.children[0]
    return b


def random_interpretation(literals: set[str]) -> Interpretation:
    return {l: random.random() >= 0.5 for l in literals}


if __name__ == "__main__":
    interpretation = {"a": True, "b": False, "c": False}

    a = Literal("a")
    b = Literal("b")
    c = Literal("c")
    func = OR([AND([a, b]), c])
    fn1 = AND([a, b])
    print(func(interpretation))
    print(func)
