import copy
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Mapping, MutableMapping, Sequence, Set
from graphlib import TopologicalSorter
from typing import Dict, Generic, TypeVar

Val = TypeVar("Val")


class Node(ABC, Generic[Val]):
    def __init__(self, key: str, ins: Set[str]) -> None:
        self.key = key
        self.ins = ins

    @abstractmethod
    def __call__(self, vars: Mapping[str, Val]) -> Val:
        pass

    def __str__(self) -> str:
        return f"Node({self.key})"


class Graph(ABC, Generic[Val]):
    def __init__(self, nodes: Sequence[Node[Val]]) -> None:
        self.nodes: Dict[str, Node[Val]] = {}
        for node in nodes:
            self.nodes[node.key] = node

        # perform various checks on the validity of the nodes data structure
        # uniqueness check of names
        assert len(self.nodes) == len(nodes), "Multiple nodes with the same name found."

        self.keys = set(self.nodes.keys())

        in_keys = set(in_key for node in self.nodes.values() for in_key in node.ins)

        # uniqueness of the target node
        self.out_keys = [key for key in self.keys if key not in in_keys]

        # sort nodes by order of execution
        d = {key: self.nodes[key].ins for key in self.keys}
        sorter = TopologicalSorter(d)
        order = sorter.static_order()

        self.input_vars = in_keys.difference(self.keys)
        self.keys = [key for key in order if key not in self.input_vars]

    def __call__(self, vars: MutableMapping[str, Val]) -> Val:
        vars = copy.copy(vars)
        for name in self.keys:
            node = self.nodes[name]
            vars[name] = node(vars)
            if name == "target":
                return vars[name]
        raise ValueError("Unaccessible code.")

    def __str__(self) -> str:
        return (
            "Graph[\n\t"
            + "\n\t".join(str(self.nodes[name]) for name in self.keys)
            + "\n]"
        )

    def topological_order(self) -> Iterable[str]:
        return self.keys

    def target_vars(self) -> Collection[str]:
        return self.out_keys
