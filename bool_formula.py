from __future__ import annotations
from enum import Enum
from typing import Any, Dict


Interpretation = Dict[str, bool]


class Op(Enum):
    AND = "&"
    OR = "|"


class Boolean:
    def __init__(self) -> None:
        pass

    def __call__(self, interpretation: Interpretation) -> bool:
        return False
    
    def all_literals(self) -> set[str]:
        return set()


class Constant(Boolean):
    def __init__(self, val: bool) -> None:
        self.val = val

    def __str__(self) -> str:
        return "T" if self.val else "F"

    def __call__(self, interpretation: Interpretation) -> bool:
        return self.val
    
    def all_literals(self) -> set[str]:
        return set()

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Constant) and __value.val == self.val


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


class Func(Boolean):
    def __init__(self, bin_op: Op, children: list[Boolean]) -> None:
        self.bin_op = bin_op
        self.children = children

    def __str__(self) -> str:
        op = str(self.bin_op.value)
        return f"({f' {op} '.join([str(c) for c in self.children])})"

    def __call__(self, interpretation: Interpretation) -> Any:
        temp = [f(interpretation) for f in self.children]
        if self.bin_op == Op.AND:
            return all(temp)
        return any(temp)
    
    def all_literals(self) -> set[str]:
        return set().union(*(f.all_literals() for f in self.children))


if __name__ == "__main__":
    interpretation = {"a": True, "b": False, "c": False}

    a = Literal("a")
    b = Literal("b")
    c = Literal("c")
    func = Func(Op.OR, [Func(Op.AND, [a, b]), c])
    fn1 = Func(Op.AND, [a, b])
    print(func(interpretation))
    print(func)
