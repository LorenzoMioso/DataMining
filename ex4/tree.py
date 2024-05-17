import math
import sys
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append("..")

from ex4.util import count_labels, parse_dataset

# DATASET_PATH = "../datasets/activity.txt"  # 35 lines
# DATASET_PATH = "../datasets/question.txt"  # 1730 lines
DATASET_PATH = "../datasets/epitope.txt"  # 2392 lines
# DATASET_PATH = "../datasets/gene.txt"  # 2942 lines
# DATASET_PATH = "../datasets/robot.txt"  # 4302 lines


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

    def weighted_frequency(y):
        return np.sum([W[j] for j in I_f if Y[j] == y])

    # a leaf node with the class that has the most weighted frequency
    tree.false_child = Leaf(max([1, -1], key=weighted_frequency))

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
        def weighted_frequency_tc(y):
            return np.sum(
                [
                    W[j]
                    for j, v in P_t.items()
                    if Y[j] == y and values[i - 1] < v <= values[i]
                ]
            )

        cnode.true_child = Leaf(max([1, -1], key=weighted_frequency_tc))

        if i < len(values) - 2:
            cnode.false_child = ValueNode()
            cnode = cnode.false_child
        else:

            # sometimes could be the same as the true child,
            def weighted_frequency_fc(y):
                return np.sum(
                    [
                        W[j]
                        for j, v in P_t.items()
                        if Y[j] == y and values[i] < v <= values[i + 1]
                    ]
                )

            cnode.false_child = Leaf(max([1, -1], key=weighted_frequency_fc))

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


def calculate_tree_pair(params, W, VT, X, Y):
    l, vt = params
    psi = TreePair(W, VT, X, Y, l, vt)
    return psi, weighted_sum(psi, W, VT, X, Y)


def weighted_sum(psi, W, VT, X, Y):
    predictions = np.array([predict(psi, (VT[i], X[i])) for i in range(len(W))])
    weighted_predictions = W * predictions * Y
    return np.sum(weighted_predictions)


def Best_tree(W, VT, X, Y, run_parallel=False) -> EventNode:
    assert (
        len(W) == len(VT) == len(X) == len(Y)
    ), f"Input data must have the same length, lengths are {len(W)}, {len(VT)}, {len(X)}, {len(Y)}"

    # Use a set to avoid duplicate pairs
    candidate_pairs = set()

    for j in range(len(W)):
        s = X[j]
        vt = VT[j]
        for si in s:
            candidate_pairs.add((si[1], si[0] - vt))

    def secont_element(pair):
        return pair[1]

    # Convert set to list and sort by the second element of the tuples
    candidate_pairs = sorted(candidate_pairs, key=secont_element)

    if run_parallel:
        with Pool(16) as p:
            results = p.starmap(
                calculate_tree_pair, [(pair, W, VT, X, Y) for pair in candidate_pairs]
            )
    else:
        results = [calculate_tree_pair(pair, W, VT, X, Y) for pair in candidate_pairs]

    best_psi, _ = max(results, key=secont_element)

    return best_psi


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
    dataset = parse_dataset(DATASET_PATH, max_items=10000, gen_tuple=count_labels)
    X = dataset[:, 0]
    Y = dataset[:, 1]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    W = np.ones(len(X_train)) / len(X_train)
    VT = np.zeros(len(X_train))
    l = "10"
    d = 10

    # tree = TreePair(W, VT, X_train, Y_train, l, d)
    # print("Tree pair")
    # print_tree(tree)

    tree = Best_tree(W, VT, X_train, Y_train, run_parallel=True)
    print("Best tree")
    print_tree(tree)

    # test the tree
    correct = 0
    for i, x in enumerate(X_test):
        y = Y_test[i]
        p = predict(tree, (0, x))
        # print(f"Predicted {p}, expected {y}")
        if p == y:
            correct += 1
    print(f"Accuracy: {correct / len(X_test)}")


if __name__ == "__main__":
    main()
