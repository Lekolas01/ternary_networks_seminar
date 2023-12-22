import copy
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence, Set
from graphlib import TopologicalSorter
from typing import Dict, Generic, TypeVar

Val = TypeVar("Val")
Key = TypeVar("Key")


class Node(ABC, Generic[Key, Val]):
    def __init__(
        self,
        name: Key,
        ins: Set[Key] = set(),
    ) -> None:
        self.name = name
        self.ins = ins

    @abstractmethod
    def __call__(self, var_setting: Dict[Key, Val]) -> Val:
        pass


class NodeGraph(ABC, Generic[Key, Val]):
    def __init__(self, nodes: Sequence[Node[Key, Val]]) -> None:
        self.nodes: Dict[Key, Node[Key, Val]] = {}
        for node in nodes:
            self.nodes[node.name] = node

        # perform various checks on the validity of the nodes data strucutre
        # uniqueness check of names
        assert len(self.nodes) == len(nodes), "Multiple nodes with the same name found."

        self.names = set(self.nodes.keys())

        in_names = set(in_name for node in self.nodes.values() for in_name in node.ins)
        self.input_vars = in_names.difference(self.names)

        output_names = [name for name in self.names if name not in in_names]
        assert len(output_names) == 1, "Only one output node may exist."
        self.out_name = output_names[0]

        # introduce topological order of the nodes
        d = {key: self.nodes[key].ins for key in self.names}
        sorter = TopologicalSorter(d)
        order = sorter.static_order()
        self.names = [name for name in order if name not in self.input_vars]

    def __call__(self, var_setting: Dict[Key, Val]) -> Val:
        var_setting = copy.copy(var_setting)
        for name in self.names:
            node = self.nodes[name]
            var_setting[name] = node(var_setting)
        return var_setting[self.out_name]

    def topological_order(self) -> Iterable[Key]:
        return self.names
