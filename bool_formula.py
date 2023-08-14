from __future__ import annotations
from enum import Enum
from typing import Any


class Op(Enum):
    AND = "&"
    OR = "|"


class Boolean:
    def __init__(self) -> None:
        pass

    def __call__(self, var: dict[str, bool]) -> bool:
        return False


class Constant(Boolean):
    def __init__(self, val: bool) -> None:
        self.val = val

    def __str__(self) -> str:
        return "T" if self.val else "F"

    def __call__(self, var: dict[str, bool]) -> Any:
        return self.val

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Constant) and __value.val == self.val


class Literal(Boolean):
    def __init__(self, value: str, positive=True) -> None:
        assert value is not None
        self.name = value
        self.positive = positive

    def __str__(self) -> str:
        return f"{'!' if not self.positive else ''}{str(self.name)}"

    def __call__(self, vars: dict[str, bool]) -> Any:
        ans = vars[self.name]
        if not self.positive:
            ans = not ans
        return ans


class Func(Boolean):
    def __init__(self, bin_op: Op, children: list[Boolean]) -> None:
        self.bin_op = bin_op
        self.children = children

    def __str__(self) -> str:
        op = str(self.bin_op.value)
        return f"({f' {op} '.join([str(c) for c in self.children])})"

    def eval(self, vals: dict[str, bool]) -> Any:
        temp = [c(vals) for c in self.children]
        if self.bin_op == Op.AND:
            return all(temp)
        return any(temp)


if __name__ == "__main__":
    # tree1 = Tree("(a & b) | c")
    instantiations = {"a": True, "b": False, "c": False}

    a = Literal("a")
    b = Literal("b")
    c = Literal("c")
    func = Func(Op.OR, [Func(Op.AND, [a, b]), c])
    fn1 = Func(Op.AND, [a, b])
    print(func(instantiations))
    print(func)
