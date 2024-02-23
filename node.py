from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence, Set
from copy import deepcopy
from graphlib import TopologicalSorter
from typing import AbstractSet, Dict, Generic, Self, TypeVar

import numpy as np


class Node(ABC):
    def __init__(self, key: str, ins: Set[str]) -> None:
        self.key = key
        self.ins = ins

    @abstractmethod
    def __call__(self, vars: Mapping[str, np.ndarray]) -> np.ndarray:
        pass

    def __str__(self) -> str:
        return f"Node({self.key})"


class Graph(ABC):
    def __init__(self, nodes: Sequence[Node]) -> None:
        self.nodes: dict[str, Node] = {}
        self.in_keys = set()
        self.out_keys = set()

        for node in nodes:
            self.add(node)

    def __call__(self, vars: MutableMapping[str, np.ndarray]) -> np.ndarray:
        order = self.topological_order()
        for key in order:
            vars[key] = self.nodes[key](vars)
            if key == "target":
                return vars[key]
        raise ValueError("Could not find a node with key 'target'.")

    def __str__(self) -> str:
        order = self.topological_order()
        ans = "Graph[\n\t" + "\n\t".join(str(self.nodes[key]) for key in order) + "\n]"
        return ans

    def __getitem__(self, key: str) -> Node:
        return self.nodes[key]

    def ins(self) -> Dict[str, Set[str]]:
        """
        A Mapping of nodes to a set of all nodes that point to the node
        """
        return {key: node.ins for key, node in self.nodes.items()}

    def outs(self) -> Dict[str, Set[str]]:
        """
        Same as graph_ins, except that the set contains all nodes that the node points towards.
        """
        inverse = {}
        graph_ins = self.ins()
        for k, v in graph_ins.items():
            for x in v:
                inverse.setdefault(x, {})[k] = x
        return inverse

    def topological_order(self) -> list[str]:
        # sort nodes by order of execution
        sorter = TopologicalSorter(self.ins())
        order = sorter.static_order()
        order = filter(lambda key: key not in self.in_keys, order)
        return list(order)

    def add(self, n: Node):
        self.nodes[n.key] = n

        if n.key in self.in_keys:
            # adding a new node can delete an in_key (if the in_key with the same name exists)
            self.in_keys.remove(n.key)
        else:
            # adding a new node can create new out_key (if no node has this as in)
            self.out_keys.add(n.key)

        for in_key in n.ins:
            if in_key in self.nodes:
                # adding a new node can delete out_keys (for each in node that is in the graph)
                self.out_keys.discard(in_key)
            else:
                # adding a new node can create new in_keys (for each in node that is not in the graph)
                self.in_keys.add(in_key)

    def debug(self) -> None:
        print(f"{self.nodes = }")
        print(f"{self.in_keys = }")
        print(f"{self.out_keys = }")
