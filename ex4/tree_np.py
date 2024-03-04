# rewrite the tree file using numpy

import numpy as np
import pandas as pd
from util import *

DATASET_PATH = "../datasets/activity.txt"
# DATASET_PATH = "../datasets/epitope.txt"
# DATASET_PATH = "../datasets/gene.txt"


class TreeNode:

    def __init__(self):
        None

    def print(self, indent=0, prefix=""):
        if isinstance(self, EventNode):
            print("  " * indent + f"EventNode('{self.l}', {self.d})")
            if self.true_child:
                self.true_child.print(indent + 1, "T")
            if self.false_child:
                self.false_child.print(indent + 1, "F")
        elif isinstance(self, ValueNode):
            print("  " * indent + f"-{prefix}: ValueNode({self.v})")
            if self.true_child:
                self.true_child.print(indent + 1, "T")
            if self.false_child:
                self.false_child.print(indent + 1, "F")
        elif isinstance(self, Leaf):
            print("  " * indent + f"-{prefix}: Leaf({self.y})")
        else:
            raise ValueError(f"Unknown type {type(self)}")

    def predict(self, x):
        vt, s_x = x  # s_x is a list of tuples (d, l, v)
        if isinstance(self, EventNode):
            # if any((l == PSI.l and d - vt <= PSI.d) for d, l, v in s_x)
            if np.where((s_x[:, 1] == self.l) & (s_x[:, 0] - vt <= self.d))[0].size > 0:
                return self.true_child.predict(x)
            else:
                return self.false_child.predict(x)
        elif isinstance(self, ValueNode):
            if s_x[0][2] > self.v:
                return self.true_child.predict(x)
            else:
                return self.false_child.predict(x)
        elif isinstance(self, Leaf):
            return self.y

    def consume(self, x):
        assert isinstance(self, EventNode)
        vt, s_x = x
        pass


class EventNode(TreeNode):
    def __init__(self, l, d, true_child=None, false_child=None):
        super().__init__()
        self.l = l
        self.d = d
        self.true_child = true_child
        self.false_child = false_child

    def __repr__(self):
        return f"EventNode({self.l}, {self.d})"


class ValueNode(TreeNode):
    def __init__(self, v=None, true_child=None, false_child=None):
        super().__init__()
        self.v = v
        self.true_child = true_child
        self.false_child = false_child

    def __repr__(self):
        return f"ValueNode({self.v})"


class Leaf(TreeNode):
    def __init__(self, y):
        super().__init__()
        self.y = y

    def __repr__(self):
        return f"Leaf({self.y})"


tree = EventNode("x0", 1)
tree.true_child = ValueNode(1)
tree.false_child = Leaf(-1)
tree.true_child.true_child = Leaf(1)
tree.true_child.false_child = Leaf(-1)

tree.print()
