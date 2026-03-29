#!/usr/bin/env python3
"""Bayesian network with conditional probability tables."""
import sys
from collections import defaultdict

class BayesNode:
    def __init__(self, name, parents=None):
        self.name, self.parents = name, parents or []
        self.cpt = {}  # tuple of parent values -> P(True)
    def set_prob(self, parent_values, prob):
        self.cpt[tuple(parent_values)] = prob
    def prob(self, value, parent_values):
        p = self.cpt.get(tuple(parent_values), 0.5)
        return p if value else 1 - p

class BayesNet:
    def __init__(self): self.nodes = {}; self.order = []
    def add_node(self, node):
        self.nodes[node.name] = node; self.order.append(node.name)
    def joint_prob(self, assignment):
        p = 1.0
        for name in self.order:
            node = self.nodes[name]
            parent_vals = tuple(assignment[pn] for pn in node.parents)
            p *= node.prob(assignment[name], parent_vals)
        return p
    def enumerate_ask(self, query_var, evidence):
        hidden = [v for v in self.order if v != query_var and v not in evidence]
        def enum(vars_left, assignment):
            if not vars_left: return self.joint_prob(assignment)
            var = vars_left[0]; rest = vars_left[1:]
            total = 0
            for val in [True, False]:
                assignment[var] = val
                total += enum(rest, assignment)
            del assignment[var]
            return total
        probs = {}
        for qval in [True, False]:
            a = dict(evidence); a[query_var] = qval
            probs[qval] = enum(hidden, a)
        total = sum(probs.values())
        return {k: v / total for k, v in probs.items()}

def main():
    if len(sys.argv) < 2: print("Usage: bayes_net.py <demo|test>"); return
    if sys.argv[1] == "test":
        net = BayesNet()
        rain = BayesNode("Rain"); rain.set_prob((), 0.2)
        sprinkler = BayesNode("Sprinkler", ["Rain"])
        sprinkler.set_prob((True,), 0.01); sprinkler.set_prob((False,), 0.4)
        wet = BayesNode("Wet", ["Rain", "Sprinkler"])
        wet.set_prob((True, True), 0.99); wet.set_prob((True, False), 0.8)
        wet.set_prob((False, True), 0.9); wet.set_prob((False, False), 0.0)
        net.add_node(rain); net.add_node(sprinkler); net.add_node(wet)
        jp = net.joint_prob({"Rain": True, "Sprinkler": False, "Wet": True})
        assert 0 < jp < 1
        result = net.enumerate_ask("Rain", {"Wet": True})
        assert 0 < result[True] < 1
        assert abs(result[True] + result[False] - 1.0) < 0.001
        # Rain more likely if wet
        prior = 0.2
        assert result[True] > prior  # observing wet increases rain probability
        print("All tests passed!")
    else:
        net = BayesNet()
        a = BayesNode("A"); a.set_prob((), 0.3)
        b = BayesNode("B", ["A"]); b.set_prob((True,), 0.9); b.set_prob((False,), 0.2)
        net.add_node(a); net.add_node(b)
        print(f"P(A|B=True) = {net.enumerate_ask('A', {'B': True})}")

if __name__ == "__main__": main()
