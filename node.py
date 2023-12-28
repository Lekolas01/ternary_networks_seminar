import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, MutableMapping, Sequence, Set
from graphlib import TopologicalSorter
from typing import Dict, Generic, TypeVar

Val = TypeVar("Val")
Key = TypeVar("Key")


class Node(ABC, Generic[Key, Val]):
    def __init__(
        self,
        key: Key,
        ins: Set[Key] = set(),
    ) -> None:
        self.key = key
        self.ins = ins

    @abstractmethod
    def __call__(self, vars: Mapping[Key, Val]) -> Val:
        pass

    def __repr__(self) -> str:
        return f"Node({self.key})"

    def __str__(self) -> str:
        return f"{self.key} <- [{', '.join(str(key) for key in self.ins)}]"


class NodeGraph(ABC, Generic[Key, Val]):
    def __init__(self, nodes: Sequence[Node[Key, Val]]) -> None:
        self.nodes: Dict[Key, Node[Key, Val]] = {}
        for node in nodes:
            self.nodes[node.key] = node

        # perform various checks on the validity of the nodes data strucutre
        # uniqueness check of names
        assert len(self.nodes) == len(nodes), "Multiple nodes with the same name found."

        self.keys = set(self.nodes.keys())

        in_keys = set(in_key for node in self.nodes.values() for in_key in node.ins)
        self.input_vars = in_keys.difference(self.keys)

        output_keys = [key for key in self.keys if key not in in_keys]
        assert len(output_keys) == 1, "Only one output node may exist."
        self.out_key = output_keys[0]

        # order nodes in order of execution
        d = {key: self.nodes[key].ins for key in self.keys}
        sorter = TopologicalSorter(d)
        order = sorter.static_order()
        self.keys = [key for key in order if key not in self.input_vars]

    def __call__(self, vars: MutableMapping[Key, Val]) -> Val:
        vars = copy.copy(vars)
        for name in self.keys:
            node = self.nodes[name]
            vars[name] = node(vars)
        return vars[self.out_key]

    def __str__(self) -> str:
        return (
            "Graph[\n\t"
            + "\n\t".join(str(self.nodes[name]) for name in self.keys)
            + "\n]"
        )

    def topological_order(self) -> Iterable[Key]:
        return self.keys
