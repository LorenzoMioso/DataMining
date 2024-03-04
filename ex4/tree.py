import functools
import math
from collections import OrderedDict

import numba as nb
import numpy as np
from numba import deferred_type, float32, float64, int8, jit, njit, optional, types
from numba.experimental import jitclass
from util import parse_dataset

DATASET_PATH = "../datasets/activity.txt"
# DATASET_PATH = "../datasets/epitope.txt"
# DATASET_PATH = "../datasets/gene.txt"


LEAF = 0
VALUE = 1
EVENT = 2

node_type = deferred_type()
spec = [
    ("type", int8),
    ("y", optional(int8)),
    ("v", optional(float32)),
    ("l", optional(types.unicode_type)),
    ("d", optional(float32)),
    ("true_child", optional(node_type)),
    ("false_child", optional(node_type)),
]


@jitclass(spec)
class Node:
    def __init__(self, type=LEAF, y=None, v=None, l=None, d=None):
        self.true_child = None
        self.false_child = None
        self.type = type
        self.y = y
        self.v = v
        self.l = l
        self.d = d


node_type.define(Node.class_type.instance_type)


@njit
def print_tree(node: Node, indent=0, prefix=""):
    if node.type == EVENT:
        print("  " * indent + f"EventNode('{node.l}', {node.d})")
        if node.true_child:
            print_tree(node.true_child, indent + 1, "T")
        if node.false_child:
            print_tree(node.false_child, indent + 1, "F")
    elif node.type == VALUE:
        print("  " * indent + f"-{prefix}: ValueNode({node.v})")
        if node.true_child:
            print_tree(node.true_child, indent + 1, "T")
        if node.false_child:
            print_tree(node.false_child, indent + 1, "F")
    elif node.type == LEAF:
        print("  " * indent + f"-{prefix}: Leaf({node.y})")
    else:
        raise ValueError(f"Unknown type {type(node)}")


@njit
def exist_event(s, vt, l, d):
    # print(f"call to exist_event with s = {s}, vt = {vt}, l = {l}, d = {d}")
    # implements ∃i(s[i].l = l, s[i].vt - vt <= d)
    for di, li, _ in s:
        if li == l and di - vt <= d:
            return True
    return False


@njit
def min_label_index(s, l):
    # implements min{i | s[i].l = l}
    for i, si in enumerate(s):
        if si[1] == l:
            return i
    return None


@njit
def best_class_by_value_true_child(
    W,
    Y,
    P_t,
    values,
    i,
):
    # true child is a leaf node with the class that has the most weighted frequency
    keys = [1, -1]
    best_key = keys[0]
    best_val = -math.inf
    for key in keys:
        val = sum(
            [
                W[int(j)]
                for j, v in P_t
                if Y[int(j)] == key and values[i - 1] < v <= values[i]
            ]
        )
        if val > best_val:
            best_val = val
            best_key = key

    return best_key


@njit
def best_class(
    W,
    Y,
    I_f,
):
    # true child is a leaf node with the class that has the most weighted frequency

    keys = [1, -1]
    best_key = keys[0]
    best_val = -math.inf
    for key in keys:
        val = sum([W[i] for i in I_f if Y[i] == key])
        if val > best_val:
            best_val = val
            best_key = key

    return best_key


@njit
def best_class_by_value_false_child(
    W,
    Y,
    P_t,
    values,
    i,
):
    # true child is a leaf node with the class that has the most weighted frequency
    keys = [1, -1]
    best_key = keys[0]
    best_val = -math.inf
    for key in keys:
        val: float = sum(
            [
                W[int(j)]
                for j, v in P_t
                if Y[int(j)] == key and values[i] < v <= values[i + 1]
            ]
        )
        if val > best_val:
            best_val = val
            best_key = key

    return best_key


@njit
def weighted_frequency(W, Y, I_f, y):
    return sum([W[j] for j in I_f if Y[j] == y])


@jit
def TreePair(W, VT, X, Y, l, d) -> Node:
    assert len(W) == len(VT) == len(X) == len(Y), "Input data must have the same lenght"
    n = len(W)
    tree = Node(EVENT, None, None, l, d)
    # indexes of all sequences that satisfy the event condition
    I_t: set[int] = set([j for j in range(len(W)) if exist_event(X[j], VT[j], l, d)])
    # indexes of all sequences that do not satisfy the event condition
    I_f: set[int] = set(range(n)).difference(I_t)

    # a leaf node with the class that has the most weighted frequency
    if I_f:
        tree.false_child = Node(LEAF, best_class(W, Y, I_f), None, None, None)
    else:
        tree.false_child = Node(LEAF, 1, None, None, None)

    # for each true sequence, save the value of the first tuple in the sequence with label l
    P_t = set()
    for j, t in enumerate([min_label_index(X[j], l) for j in range(n)]):
        if j in I_t:
            P_t.add((float(j), float(X[j][t][2])))

    # P_t should not have None values
    assert None not in [i for i, _ in P_t], "P_t should not have None values"

    # sorted list of unique values in P_t with -inf
    values = set([-math.inf]).union(set([p[1] for p in P_t]))
    values = sorted(list(values))
    # print("values = ", values)

    tree.true_child = Node(VALUE, None, None, None, None)
    cnode = tree.true_child

    i = 1
    while True:
        cnode.v = values[i]
        # splitting by value clould results in a single class
        cnode.true_child = Node(
            LEAF, best_class_by_value_true_child(W, Y, P_t, values, i), None, None, None
        )

        if i < len(values) - 2:
            cnode.false_child = Node(type=VALUE)
            cnode = cnode.false_child
        else:
            # sometimes is the same as the true child
            cnode.false_child = Node(
                LEAF,
                best_class_by_value_false_child(W, Y, P_t, values, i),
                None,
                None,
                None,
            )

        i += 1
        if i > len(values) - 1:
            break
    return tree


@njit
def predict(PSI, x):
    # print(f"Predicting {x} with PSI {print_tree(PSI)}")
    vt, s_x = x
    assert len(x) > 0, "Input sequence must have at least one element"
    assert PSI is not None, "PSI must not be None"
    while PSI.type != LEAF:
        # print(f"inside while, PSI : {PSI}, x : {x}")
        if PSI.type == EVENT:
            PSI = (
                PSI.true_child
                if exist_event(s_x, vt, PSI.l, PSI.d)
                else PSI.false_child
            )
            if PSI is None:
                raise ValueError("PSI is None after EventNode")
        elif PSI.type == VALUE:
            PSI = PSI.true_child if s_x[0][2] == PSI.v else PSI.false_child
            if PSI is None:
                raise ValueError("PSI is None after ValueNode")
    # print(f"result : {PSI.y}, type : {type(PSI.y)}")
    return PSI.y


@njit
def consume(PSI, x):
    assert PSI.type == EVENT, "PSI must be an EventNode"
    vt, s_x = x
    i = None
    # assuming s_x is sorted
    for idx, s in enumerate(s_x):
        if vt - s[0] <= PSI.d and s[1] == PSI.l:
            i = idx
            break
    if i is not None:
        return (s_x[i][0], s_x[i + 1 :])
    else:
        return (vt, s_x)


def Best_tree(W, VT, X, Y):
    # print("call to Best_tree")
    assert (
        len(W) == len(VT) == len(X) == len(Y)
    ), f"Input data must have the same lenght, lengths are {len(W)}, {len(VT)}, {len(X)}, {len(Y)}"
    candidate_pairs = []
    n = len(W)
    for j in range(n):
        s = X[j]
        vt = VT[j]
        for si in s:
            if (si[1], si[0] - vt) not in candidate_pairs:
                candidate_pairs.append((si[1], si[0] - vt))

    PSI = [TreePair(W, VT, X, Y, l, vt) for l, vt in candidate_pairs]

    return max(
        PSI,
        key=lambda psi: sum(
            [w * (predict(psi, (VT[i], X[i])) - Y[i]) for i, w in enumerate(W)]
        ),
    )


def main():
    print("call to main")
    df = parse_dataset(DATASET_PATH)
    train = df.sample(frac=0.8)
    test = df.drop(train.index)

    W = np.ones(len(train)) / len(train)
    VT = np.zeros(len(train))
    X = train["s"].tolist()
    Y = train["y"].tolist()
    l = "7"
    d = 18

    # tree_pair = TreePair(W, VT, X, Y, l, d)
    # print("Tree pair")
    # print_tree(tree_pair)

    tree = Best_tree(W, VT, X, Y)
    print("Best tree")
    print_tree(tree)

    # test the tree
    correct = 0
    for i, row in test.iterrows():
        x = row["s"]
        y = row["y"]
        p = predict(tree, x)
        print(f"Predicted {p}, expected {y}")
        if p == y:
            correct += 1
    print(f"Accuracy: {correct / len(test)}")


def is_valid_weak_learner():
    cycles = 100
    accuracies = []
    count_accurancy_50 = 0
    for _ in range(cycles):
        print(f"Cycle {_ + 1}")
        df = parse_dataset(DATASET_PATH)
        train = df.sample(frac=0.8)
        test = df.drop(train.index)

        W = np.ones(len(train)) / len(train)
        VT = np.zeros(len(train))
        X = train["s"].tolist()
        Y = train["y"].tolist()

        tree = Best_tree(W, VT, X, Y)

        # print("Best tree")
        # print_tree(tree)

        correct = 0
        for _, row in test.iterrows():
            x = row["s"]
            y = row["y"]
            p = predict(tree, (0, x))
            if p == y:
                correct += 1
        accuracies.append(correct / len(test))
        if accuracies[-1] >= 0.5:
            count_accurancy_50 += 1

    print(f"Mean accuracy: {np.mean(accuracies)}")
    print(f"Percentage of times with accuracy over 50%: {count_accurancy_50 / cycles}")
    print(f"Standard deviation: {np.std(accuracies)}")

    # Results with mechanism to make a real tree
    ### activity.txt (100 itrs) ###
    # Mean accuracy: 0.54
    # Percentage of times with accuracy over 50%: 0.62
    # Standard deviation: 0.18
    ### epitope.txt (100 itrs) ###
    # Mean accuracy: 0.558
    # Percentage of times with accuracy over 50%: 1
    # Standard deviation: 0.018
    ### gene.txt (100 itrs) ###
    # Mean accuracy: 0.491
    # Percentage of times with accuracy over 50%: 0.37
    # Standard deviation: 0.019

    # Results with exercise tree
    ### activity.txt (100 itrs) ###
    # Mean accuracy: 0.61
    # Percentage of times with accuracy over 50%: 0.72
    # Standard deviation: 0.178
    ### epitope.txt (100 itrs) ###
    # Mean accuracy: 0.560
    # Percentage of times with accuracy over 50%: 1
    # Standard deviation: 0.0191
    ### gene.txt (100 itrs) ###
    # Mean accuracy: 0.60
    # Percentage of times with accuracy over 50%: 0.95
    # Standard deviation: 0.03


if __name__ == "__main__":
    is_valid_weak_learner()
    # main()
