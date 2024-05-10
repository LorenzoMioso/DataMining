import math
import sys
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np

sys.path.append("..")

from ex4.util import count_labels, parse_dataset

# DATASET_PATH = "../datasets/activity.txt"  # 35 lines
DATASET_PATH = "../datasets/question.txt"  # 1730 lines
# DATASET_PATH = "../datasets/epitope.txt"  # 2392 lines
# DATASET_PATH = "../datasets/gene.txt"  # 2942 lines
# DATASET_PATH = "../datasets/robot.txt" # 4302 lines


class EventNode:
    def __init__(self, l, d, true_child=None, false_child=None):
        self.l = l
        self.d = d
        self.true_child = true_child
        self.false_child = false_child

    def __repr__(self):
        return f"EventNode({self.l}, {self.d})"


class ValueNode:
    def __init__(self, v=None, true_child=None, false_child=None):
        self.v = v
        self.true_child = true_child
        self.false_child = false_child

    def __repr__(self):
        return f"ValueNode({self.v})"


class Leaf:
    def __init__(self, y):
        self.y = y

    def __repr__(self):
        return f"Leaf({self.y})"


class SeqTree:
    event_node: EventNode

    def __init__(self):
        pass

    def fit(self, W, VT, X, Y, run_parallel=False):
        self.event_node = Best_tree(W, VT, X, Y, run_parallel=run_parallel)
        return self

    def predict(self, x):
        return predict(self.event_node, x)

    def consume(self, x):
        return consume(self.event_node, x)

    def print(self):
        print_tree(self.event_node)


def exist_event(s, vt, l, d):
    # implements ∃i(s[i].l = l, s[i].vt - vt <= d)
    for di, li, _ in s:
        if li == l and di - vt <= d:
            return True
    return False


def min_label_index(s, l):
    # implements min{i | s[i].l = l}
    for i, si in enumerate(s):
        if si[1] == l:
            return i
    return None


def TreePair(W, VT, X, Y, l, d) -> EventNode:
    assert len(W) == len(VT) == len(X) == len(Y), "Input data must have the same length"
    n = len(W)
    tree = EventNode(l, d)
    I_t = set()
    I_f = set(range(len(W)))

    for j in range(n):
        if exist_event(X[j], VT[j], l, d):
            I_t.add(j)
        else:
            I_f.discard(j)

    # a leaf node with the class that has the most weighted frequency
    tree.false_child = Leaf(
        max([1, -1], key=lambda y: np.sum([W[j] for j in I_f if Y[j] == y]))
    )

    # for each true sequence, save the value of the first tuple in the sequence with label l
    P_t = {
        j: X[j][t]["v"]
        for j, t in enumerate([min_label_index(X[j], l) for j in range(n)])
        if j in I_t
    }

    values = [-math.inf] + sorted(set(P_t.values()))

    tree.true_child = ValueNode()
    cnode = tree.true_child

    i = 1
    while True:
        cnode.v = values[i]

        # could be that splitting by value results in a single class
        # true child is a leaf node with the class that has the most weighted frequency
        cnode.true_child = Leaf(
            max(
                [1, -1],
                key=lambda y: np.sum(
                    [
                        W[j]
                        for j, v in P_t.items()
                        if Y[j] == y and values[i - 1] < v <= values[i]
                    ]
                ),
            )
        )

        if i < len(values) - 2:
            cnode.false_child = ValueNode()
            cnode = cnode.false_child
        else:

            # sometimes could be the same as the true child,
            cnode.false_child = Leaf(
                max(
                    [1, -1],
                    key=lambda y: np.sum(
                        [
                            W[j]
                            for j, v in P_t.items()
                            if Y[j] == y and values[i] < v <= values[i + 1]
                        ]
                    ),
                )
            )

        i += 1
        if i > len(values) - 1:
            break
    return tree


def predict(PSI, x):
    vt, s_x = x
    assert len(x) > 0, "Input sequence must have at least one element"
    assert PSI is not None, "PSI must not be None"
    while not isinstance(PSI, Leaf):
        if isinstance(PSI, EventNode):
            PSI = (
                PSI.true_child
                if any((l == PSI.l and d - vt <= PSI.d) for d, l, v in s_x)
                else PSI.false_child
            )
        elif isinstance(PSI, ValueNode):
            PSI = PSI.true_child if s_x[0][2] == PSI.v else PSI.false_child
    return PSI.y


def consume(PSI, x) -> Tuple[int, List[Tuple[int, str, int]]]:
    assert isinstance(PSI, EventNode)
    vt, s_x = x
    i = None
    for idx, (d, l, _) in enumerate(s_x):
        if vt - d <= PSI.d and l == PSI.l:
            i = idx
            break
    if i is not None:
        return (s_x[i]["d"], s_x[i + 1 :])
    else:
        return (vt, s_x)


def Best_tree(W, VT, X, Y, run_parallel=False) -> EventNode:
    assert (
        len(W) == len(VT) == len(X) == len(Y)
    ), f"Input data must have the same length, lengths are {len(W)}, {len(VT)}, {len(X)}, {len(Y)}"
    candidate_pairs = []

    for j in range(len(W)):
        s = X[j]
        vt = VT[j]
        for si in s:
            if (si[1], si[0] - vt) not in candidate_pairs:
                candidate_pairs.append((si[1], si[0] - vt))

    # sort the candidate pairs by incresing order of the second element.
    # The intention is to reduce the consumption of the sequences
    candidate_pairs.sort(key=lambda x: x[1])

    PSI = []
    if run_parallel:
        with Pool(16) as p:
            PSI = p.starmap(
                TreePair, [(W, VT, X, Y, l, vt) for l, vt in candidate_pairs]
            )
    else:
        PSI = [TreePair(W, VT, X, Y, l, vt) for l, vt in candidate_pairs]

    return max(
        PSI,
        key=lambda psi: np.sum(
            [w * (predict(psi, (VT[i], X[i])) * Y[i]) for i, w in enumerate(W)]
        ),
    )


def print_tree(root, indent=0, prefix=""):
    if isinstance(root, EventNode):
        print("  " * indent + f"EventNode('{root.l}', {root.d})")
        if root.true_child:
            print_tree(root.true_child, indent + 1, "T")
        if root.false_child:
            print_tree(root.false_child, indent + 1, "F")
    elif isinstance(root, ValueNode):
        print("  " * indent + f"-{prefix}: ValueNode({root.v})")
        if root.true_child:
            print_tree(root.true_child, indent + 1, "T")
        if root.false_child:
            print_tree(root.false_child, indent + 1, "F")
    elif isinstance(root, Leaf):
        print("  " * indent + f"-{prefix}: Leaf({root.y})")
    else:
        raise ValueError(f"Unknown type {type(root)}")


def main():
    df = parse_dataset(DATASET_PATH, max_items=10000, gen_tuple=count_labels)
    train = df.sample(frac=0.8)
    test = df.drop(train.index)

    W = np.ones(len(train)) / len(train)
    VT = np.zeros(len(train))
    X = train["s"].to_numpy()
    Y = train["y"].to_numpy()
    l = "5"
    d = 28

    tree = TreePair(W, VT, X, Y, l, d)
    print("Tree pair")
    print_tree(tree)

    # tree = Best_tree(W, VT, X, Y)
    # print("Best tree")
    # print_tree(tree)

    # test the tree
    correct = 0
    for i, row in test.iterrows():
        x = row["s"]
        y = row["y"]
        p = predict(tree, (0, x))
        # print(f"Predicted {p}, expected {y}")
        if p == y:
            correct += 1
    print(f"Accuracy: {correct / len(test)}")


if __name__ == "__main__":
    main()
