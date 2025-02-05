import bisect
import copy
import functools
from collections.abc import Mapping, MutableMapping, Sequence
from graphlib import TopologicalSorter
from typing import Tuple

import numpy as np
import pandas as pd

from bool_formula import NOT, Constant, Knowledge, Literal
from neuron import bool_2_ch
from node import Graph, Node
from q_neuron import Perceptron, QuantizedNeuronGraph
from utilities import flatten


class DpNode:
    def __init__(self, key: str, min_thr: float, max_thr: float) -> None:
        self.key = key
        self.min_thr = min_thr
        self.max_thr = max_thr
        self.mean = (min_thr + max_thr) / 2

    def __repr__(self) -> str:
        return f"DpNode({self.key}, {self.min_thr}, {self.max_thr})"


class Dp:
    def __init__(self, n_vars: int) -> None:
        assert 0 <= n_vars
        self.n_vars = n_vars
        self.data: list[list[DpNode]] = [[] for _ in range(self.n_vars + 1)]

    def find(self, k: int, val: float) -> DpNode | None:
        assert k >= 0, f"k must be >= 0, but got {k}."
        if k > len(self.data):
            return None
        arr = self.data[k]
        for t in arr:
            if t.min_thr < val < t.max_thr:
                return t
        return None

    def insert(self, k: int, val: DpNode):
        # assert you don't add part of a range that is already included
        if (
            self.find(k, val.min_thr) is not None
            or self.find(k, val.max_thr) is not None
        ):
            print(f"{self.data[k] = }")
            print(f"{k = }")
            print(f"{val = }")
            raise ValueError

        bisect.insort(self.data[k], val, key=lambda x: x.min_thr)

    def __getitem__(self, key: int):
        assert 0 <= key <= self.n_vars
        return self.data[key]

    def __str__(self) -> str:
        ans = []
        for i in range(self.n_vars + 1):
            ans.append("[" + ", ".join(str(t) for t in self.data[i]) + "]")
        return "\n".join(ans)

    def __repr__(self) -> str:
        return str(self)

    def __len__(self):
        return len(self.data)


class IfThenRule(Node):
    def __init__(
        self, key: str, ins: list[Tuple[str, bool]], val: bool | None = None
    ) -> None:
        self.key = key
        self.ins = ins
        self.val = val
        self.is_const = self.val is not None

    def __call__(self, vars: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.is_const:
            return np.array(True) if self.val else np.array(False)

        key = self.ins[0][0]
        ans = np.ones_like(vars[key], dtype=bool)
        for name, val in self.ins:
            temp = vars[name] if val else ~vars[name]
            try:
                ans = ans & temp
            except:
                raise ValueError

        return ans

    def simplify(self, knowledge: Knowledge) -> bool:
        if self.is_const:
            assert isinstance(self.val, bool)
            ans = not (self.key in knowledge)
            knowledge[self.key] = Constant(self.val)
            return ans
        changed = False
        to_delete_ins = []

        for idx, (lit, lit_val) in enumerate(self.ins):
            if lit in knowledge:
                if isinstance(knowledge[lit], Constant):
                    new_lit_val = lit_val if knowledge[lit]() else not lit_val
                    if new_lit_val:
                        # delete any positive constants
                        to_delete_ins.append(lit)
                        changed = True
                    else:
                        # if you have a negative constant, the whole rule is negative
                        self.is_const = True
                        self.val = False
                        return True
                elif isinstance(knowledge[lit], Literal):
                    self.ins[idx] = (str(knowledge[lit]), lit_val)
                elif isinstance(knowledge[lit], NOT):
                    c = knowledge[lit].child  # type: ignore
                    self.ins[idx] = (str(c), not lit_val)

                # elif isinstance(knowledge[lit], NOT):
                # self.ins[idx] = (str(knowledge[lit]), lit_val)
        self.ins = [t for t in self.ins if t[0] not in to_delete_ins]
        # if there are no literals left, i.e. every literal is positive, the whole rule is positive
        if len(self.ins) == 0:
            self.is_const = True
            self.val = True
            return True
        return changed

    def __repr__(self) -> str:
        return f"{self.key} := {self.body()}"

    def __str__(self) -> str:
        return self.__repr__()

    def body(self) -> str:
        if self.is_const:
            assert isinstance(self.val, bool)
            return bool_2_ch(self.val)
        return ", ".join(f"{'' if b else '!'}{key}" for key, b in self.ins)

    def children(self) -> list[str]:
        if self.is_const:
            return []
        return [key for key, _ in self.ins]

    def complexity(self) -> int:
        return 1 if self.is_const else len(self.ins)


class Subproblem:
    def __init__(self, key: str, rules: list[IfThenRule]):
        self.key = key
        self.rules = rules
        self.is_const = False
        self.val = False

    def __call__(self, vars: MutableMapping[str, np.ndarray]) -> np.ndarray:
        if self.is_const:
            return np.array(self.val)
        temp = [rule(vars) for rule in self.rules]
        return functools.reduce(lambda x, y: x | y, temp)

    def simplify(self, knowledge: Knowledge) -> bool:
        """
        Returns a boolean that tells whether something changed
        """
        # first, simplify each individual rule.
        # if no rule changed, the whole subproblem did not change, -> return False.
        changed = False
        for rule in self.rules:
            temp = rule.simplify(knowledge)
            changed = changed or temp

        # filter all constant F rules, as they will never trigger
        self.rules = [r for r in self.rules if not (r.is_const and not r.val)]

        # if there exists a constant T rule, the whole subproblem is T
        if any(r.is_const and r.val for r in self.rules):
            knowledge[self.key] = Constant(True)
            self.is_const, self.val = True, True

        # if there are no non-constant rules left, the whole subproblem is F
        if len(self.rules) == 0:
            knowledge[self.key] = Constant(False)
            self.is_const, self.val = True, False

        # if the subproblem only has one rule left with exactly one value,
        # add the value to the knowledge and delete the subproblem
        if len(self.rules) == 1 and len(self.rules[0].ins) == 1:
            t = self.rules[0].ins[0]
            v = Literal(t[0])
            if not t[1]:
                v = NOT(v)
            knowledge[self.key] = v

        """
        to further simplify subproblems, one should also look at relations between 
        the rules. For example, if one rule's body is a strict superset of another rule's
        body, you can delete the more specific rule, as it will only ever trigger when the 
        more general triggers anyways. In our use case however, that can not happen.
        """
        return changed

    def __str__(self) -> str:
        if self.is_const:
            return f"SP({self.key} := {bool_2_ch(self.val)})"
        return "\n\t".join([f"{self.key}\t:= {r.body()}" for r in self.rules])
        return f"SP({self.key} := {' | '.join(r.body() for r in self.rules)})"

    def __repr__(self) -> str:
        return str(self)

    def children(self) -> list[str]:
        return flatten([rule.children() for rule in self.rules if not rule.is_const])

    def complexity(self) -> int:
        return sum(rule.complexity() for rule in self.rules)


class RuleSetNeuron(Node):
    def __init__(
        self, q_neuron: Perceptron, q_ng: QuantizedNeuronGraph, simplify: bool
    ) -> None:
        self.q_neuron = q_neuron
        self.q_ng = q_ng
        self.key = q_neuron.key
        self.ins = q_neuron.ins
        self.dp = self.calc_dp()
        self.subproblems = self.to_subproblems(self.dp)
        self.target_node = self.call_order()[-1]
        self.knowledge = {}
        if simplify:
            # simplify rules
            self.simplify()

    def __call__(self, vars: MutableMapping[str, np.ndarray]) -> np.ndarray:
        vars = copy.copy(vars)
        for k in self.knowledge:
            # print(f"Set {self.knowledge[k]} to {k}.")
            if isinstance(self.knowledge[k], Constant):
                vars[k] = np.array(self.knowledge[k]())
            else:
                vars[k] = np.array(self.knowledge[k])
        for key in self.call_order():
            sp = self.subproblems[key]
            vars[key] = sp(vars)
        return vars[self.key]

    def name_gen(self):
        yield self.key
        idx = 0
        while True:
            idx += 1
            yield f"{self.key}_{idx}"

    def calc_dp(self) -> Dp:
        def to_bool_rec(k: int, threshold: float, dp: Dp) -> DpNode:
            max_sum: float = float(sum(n[1] for n in self.n_ins[k:]))
            # how much one could add by setting every variable to 1 without changing the formula
            found = dp.find(k, threshold)
            if found is not None:
                return found

            # if already positive, return True
            if threshold >= 0:
                ans = DpNode("rename_me", 0.0, float("inf"))
                dp.insert(k, ans)
                return ans
            # if you can't reach positive values, return False
            if max_sum + threshold <= 0.0:
                ans = DpNode("rename_me", float("-inf"), -max_sum)
                dp.insert(k, ans)
                return ans

            weight = self.n_ins[k][1]
            assert weight > 0

            # set to False
            n1 = to_bool_rec(k + 1, threshold, dp)
            n2 = to_bool_rec(k + 1, threshold + weight, dp)
            new_min, new_max = (
                max(n1.min_thr, n2.min_thr - weight),
                min(n1.max_thr, n2.max_thr - weight),
            )
            assert new_min <= new_max, f"{new_min} is not <= {new_max}."
            ans = DpNode("rename_me", new_min, new_max)
            dp.insert(k, ans)
            return ans

        self.bias = self.q_neuron.bias

        # adjust the weights given the y_centers of the previous layer
        # ins = list(self.ins.items())
        ins = copy.copy(self.ins)
        for key in ins:
            w = ins[key]
            if key in self.q_ng.in_keys:
                continue
            in_node = self.q_ng[key]
            assert isinstance(in_node, Perceptron)
            y_centers = in_node.y_centers
            a = y_centers[0]
            k = y_centers[1] - y_centers[0]
            self.bias += a * w  # update bias
            ins[key] *= k  # update weight

        # sort neurons by their weight
        ins = list(ins.items())
        ins = sorted(ins, key=lambda x: abs(x[1]), reverse=True)

        # remember which weights are negative and then invert all of them (needed for negative numbers)
        self.signs = [tup[1] < 0 for tup in ins]
        self.n_ins = [(tup[0], abs(tup[1])) for tup in ins]
        self.n_vars = len(self.n_ins)

        positive_weights = list(zip(self.signs, [tup[1] for tup in self.n_ins]))
        filtered_weights = list(filter(lambda tup: tup[0], positive_weights))
        self.bias_diff = sum(tup[1] for tup in filtered_weights)

        dp = Dp(len(self.n_ins))
        self.ans = to_bool_rec(0, self.bias - self.bias_diff, dp)

        # give every node in the directed bool graph a name
        names = self.name_gen()
        for k in range(self.n_vars + 1):
            for node in dp[k]:
                node.key = next(names)
        return dp

    @classmethod
    def from_q_neuron(cls, q_neuron: Perceptron):
        return RuleSetNeuron(q_neuron, True)

    def __str__(self) -> str:
        ans = (
            f"RuleSet {self.key} [\n\t"
            + "\n\t".join(str(sp) for sp in self.subproblems.values())
            + "\n]"
        )
        return ans

    def __repr__(self) -> str:
        return str(self)

    def to_subproblems(self, dp: Dp) -> dict[str, Subproblem]:
        ans: dict[str, Subproblem] = {}
        # Operator()
        # then create 1 or 2 if-then rules for each node, depending on whether it's
        # a constant or not
        for k in range(self.n_vars + 1):
            for node in dp[k]:
                if node.min_thr == float("-inf"):
                    ans[node.key] = Subproblem(
                        node.key, [IfThenRule(node.key, [], val=False)]
                    )
                elif node.max_thr == float("inf"):
                    ans[node.key] = Subproblem(
                        node.key, [IfThenRule(node.key, [], val=True)]
                    )
                else:
                    target_1 = dp.find(k + 1, node.mean)
                    if target_1 is None:
                        # print(f"{dp = }")
                        print(f"{k + 1 = }")
                        print(f"{node.mean = }")
                    if target_1 is None:
                        raise ValueError

                    target_2 = dp.find(k + 1, node.mean + self.n_ins[k][1])
                    assert target_2 is not None
                    rule1 = IfThenRule(node.key, [(target_1.key, True)])
                    rule2 = IfThenRule(
                        node.key,
                        [
                            (self.n_ins[k][0], not self.signs[k]),
                            (target_2.key, True),
                        ],
                    )
                    ans[node.key] = Subproblem(key=node.key, rules=[rule1, rule2])
        return ans

    def graph_ins(self) -> dict[str, set[str]]:
        graph_ins = {}
        keys = {key for key in self.subproblems}
        for key, sp in self.subproblems.items():
            graph_ins[key] = list(filter(lambda k: k in keys, sp.children()))
        return graph_ins

    def input_rules(self) -> set[str]:
        ans: set[str] = set()
        for sp in self.subproblems.values():
            for child in sp.children():
                ans.add(child)
        return ans

    def call_order(self) -> list[str]:
        sorter = TopologicalSorter(self.graph_ins())
        return list(sorter.static_order())

    def simplify(self) -> None:
        changed = True
        # simplify each subproblem in topological order
        while changed:
            changed = False
            for key in self.call_order():
                temp = self.subproblems[key].simplify(self.knowledge)
                changed = changed or temp
            # filter constant subproblems
            self.subproblems = {
                key: sp for key, sp in self.subproblems.items() if not sp.is_const
            }
            # filter subproblems that are not in use anymore
            all_children = set(
                flatten(sp.children() for sp in self.subproblems.values())
            )
            all_children.add(self.target_node)
            old_len = len(self.subproblems)
            self.subproblems = {
                key: sp for key, sp in self.subproblems.items() if key in all_children
            }

    def complexity(self) -> int:
        return sum(sp.complexity() for sp in self.subproblems.values())


class RuleSetGraph(Graph):
    def __init__(self, rule_set_neurons: Sequence[RuleSetNeuron]) -> None:
        super().__init__(rule_set_neurons)

    @classmethod
    def from_QNG(cls, q_ng: QuantizedNeuronGraph, simplify=True):
        prune_thr = 0.04
        # keep track of the needed rule set neurons
        needed_rule_set_neurons: set[str] = {"target"}
        rule_set_neurons = []

        q_neurons = list(q_ng.nodes.items())
        q_neurons.reverse()
        for key, q_n in q_neurons:
            if not isinstance(q_n, Perceptron):
                continue
            if key not in needed_rule_set_neurons:
                continue
            q_n.prune(prune_thr)
            rule_set_neuron = RuleSetNeuron(q_n, q_ng, simplify)
            graph_ins = rule_set_neuron.input_rules()
            rule_set_neurons.append(rule_set_neuron)
            for name in graph_ins:
                needed_rule_set_neurons.add(name)
        # rule_neurons = [
        #     RuleSetNeuron(q_n, q_ng, simplify)
        #     for key, q_n in q_ng.nodes.items()
        #     if isinstance(q_n, Perceptron) and key in needed_rule_set_neurons
        # ]
        ans = RuleSetGraph(rule_set_neurons)

        if simplify:
            ans.simplify()
        return ans

    def __repr__(self):
        return str(self)

    def complexity(self) -> int:
        return sum(
            node.complexity()
            for node in self.nodes.values()
            if isinstance(node, RuleSetNeuron)
        )

    def __call__(self, data: pd.DataFrame | dict[str, np.ndarray]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            keys = list(data.columns)
            data = {key: np.array(data[key], dtype=bool) for key in keys}
        return super().__call__(data)

    def simplify(self):

        pass
