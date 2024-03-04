import math
from typing import List, Tuple

import numpy as np
from numba import jit
from util import parse_dataset

# DATASET_PATH = "../datasets/activity.txt"
# DATASET_PATH = "../datasets/epitope.txt"
DATASET_PATH = "../datasets/gene.txt"


# TODO make the tree a class
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


def exist_event(s, vt, l, d):
    # print(f"call to exist_event with s = {s}, vt = {vt}, l = {l}, d = {d}")
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
    assert len(W) == len(VT) == len(X) == len(Y), "Input data must have the same lenght"
    n = len(W)
    tree = EventNode(l, d)
    # indexes of all sequences that satisfy the event condition
    I_t: set[int] = set([j for j in range(len(W)) if exist_event(X[j], VT[j], l, d)])
    # indexes of all sequences that do not satisfy the event condition
    I_f: set[int] = set(range(len(W))).difference(I_t)

    # print("I_t = ", I_t)
    # print("I_f = ", I_f)

    # print("classes of I_f")
    # for j in I_f:
    #    print(Y[j])

    # a leaf node with the class that has the most weighted frequency
    tree.false_child = Leaf(
        max([1, -1], key=lambda y: sum([W[j] for j in I_f if Y[j] == y]))
    )

    # print("tree.false_child = ", tree.false_child)

    # for each true sequence, save the value of the first tuple in the sequence with label l
    P_t = set(
        (j, X[j][t][2])
        for j, t in enumerate([min_label_index(X[j], l) for j in range(n)])
        if j in I_t
    )

    # print("P_t = ", P_t)

    # P_t should not have None values
    assert None not in [i for i, _ in P_t], "P_t should not have None values"

    # sorted list of unique values in P_t with -inf
    values = set([-math.inf]).union(set([p[1] for p in P_t]))
    values = sorted(list(values))
    # print("values = ", values)

    tree.true_child = ValueNode()
    cnode = tree.true_child

    i = 1
    while True:
        cnode.v = values[i]
        # print(f"cnode.v = {cnode.v}")

        # handle the case where splitting by value results in a single class
        # c = set([Y[j] for j, v in P_t if values[i - 1] < v <= values[i]])
        # if len(c) == 1:
        #    # print("single class")
        #    cnode.true_child = Leaf(list(c)[0])
        #    cnode.false_child = Leaf(list(c)[0])
        #    break

        # print("working on true_child")

        # for j, v in P_t:
        #    if values[i - 1] < v <= values[i]:
        #        print(f"j = {j}, v = {v}, Y[j] = {Y[j]}")

        # print("count of 1")
        # print(sum([1 for j, v in P_t if Y[j] == 1 and values[i - 1] < v <= values[i]]))
        # print("count of -1")
        # print(sum([1 for j, v in P_t if Y[j] == -1 and values[i - 1] < v <= values[i]]))

        # true child is a leaf node with the class that has the most weighted frequency
        cnode.true_child = Leaf(
            max(
                [1, -1],
                key=lambda y: sum(
                    [
                        W[j]
                        for j, v in P_t
                        if Y[j] == y and values[i - 1] < v <= values[i]
                    ]
                ),
            )
        )

        # print("cnode.true_child = ", cnode.true_child)

        if i < len(values) - 2:
            # print("len(values) - 2")
            cnode.false_child = ValueNode()
            cnode = cnode.false_child
        else:

            # print("working on false_child")

            # for j, v in P_t:
            #    if values[i] < v <= values[i + 1]:
            #        print(f"j = {j}, v = {v}, Y[j] = {Y[j]}")

            # print("count of 1")
            # print(
            #    sum([1 for j, v in P_t if Y[j] == 1 and values[i] < v <= values[i + 1]])
            # )
            # print("count of -1")
            # print(
            #    sum(
            #        [1 for j, v in P_t if Y[j] == -1 and values[i] < v <= values[i + 1]]
            #    )
            # )

            # is the same as the true child,
            # but true and false child should not be the same
            cnode.false_child = Leaf(
                max(
                    [1, -1],
                    key=lambda y: sum(
                        [
                            W[j]
                            for j, v in P_t
                            if Y[j] == y and values[i] < v <= values[i + 1]
                        ]
                    ),
                )
            )

            # false child should not be the same as the true child
            # if cnode.false_child.y == cnode.true_child.y:
            #    # print("false child is the same as the true child")
            #    cnode.false_child.y = -cnode.false_child.y

            # print("cnode.false_child = ", cnode.false_child)

        i += 1
        if i > len(values) - 1:
            break
    return tree


def predict(PSI, x):
    # print(f"Predicting {x} with PSI {print_tree(PSI)}")
    vt, s_x = x
    assert len(x) > 0, "Input sequence must have at least one element"
    assert PSI is not None, "PSI must not be None"
    while not isinstance(PSI, Leaf):
        # print(f"inside while, PSI : {PSI}, x : {x}")
        if isinstance(PSI, EventNode):
            PSI = (
                PSI.true_child
                if any((l == PSI.l and d - vt <= PSI.d) for d, l, v in s_x)
                else PSI.false_child
            )
            if PSI is None:
                raise ValueError("PSI is None after EventNode")
        elif isinstance(PSI, ValueNode):
            PSI = PSI.true_child if s_x[0][2] == PSI.v else PSI.false_child
            if PSI is None:
                raise ValueError("PSI is None after ValueNode")
    # print(f"result : {PSI.y}, type : {type(PSI.y)}")
    return PSI.y


def consume(PSI, x):
    # print(f"Consuming {x} with {PSI}")
    # x = (vt, s_x) , vt belongs to R^(>=0), s_x belongs to s_x belongs to (R^+ x L, x R)^*
    assert isinstance(PSI, EventNode)
    vt, s_x = x
    # print(f"vt = {vt}, s_x = {s_x}")
    i = None
    # assuming s_x is sorted
    for idx, s in enumerate(s_x):
        if vt - s[0] <= PSI.d and s[1] == PSI.l:
            i = idx
            break
    if i is not None:
        # print(f"i = {i}")
        return (s_x[i][0], s_x[i + 1 :])
    else:
        return (vt, s_x)


def Best_tree(W, VT, X, Y) -> EventNode:
    # print("call to Best_tree")
    assert (
        len(W) == len(VT) == len(X) == len(Y)
    ), f"Input data must have the same lenght, lengths are {len(W)}, {len(VT)}, {len(X)}, {len(Y)}"
    candidate_pairs: List[Tuple[str, int]] = []
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
