import math
import sys

import numpy as np

sys.path.append("..")
from ex4.tree import Best_tree, consume, predict

DATASET_PATH = "../datasets/activity.txt"  # 35 lines
# DATASET_PATH = "../datasets/question.txt"  # 1730 lines
# DATASET_PATH = "../datasets/epitope.txt"  # 2392 lines
# DATASET_PATH = "../datasets/gene.txt"  # 2942 lines
# DATASET_PATH = "../datasets/robot.txt"  # 4302 lines
ITERATIONS = 50


def fix_after_consume(X, W, Y):
    # remove empty sequences from X and adjust Z, W, Y
    n = len(W)
    empty_sequences = [i for i in range(n) if len(X[1][i]) == 0]
    new_X = ([], [])
    new_W = []
    new_Y = []
    for i in range(n):
        if i not in empty_sequences:
            new_X[0].append(X[0][i])
            new_X[1].append(X[1][i])
            new_W.append(W[i])
            new_Y.append(Y[i])

    # normalize W
    new_W = [w / sum(new_W) for w in new_W]
    return new_X, new_W, new_Y


def ada_boost(X_0: list, Y: list, iterations: int):
    t = 0
    n = len(X_0)
    X: List[Tuple[List[int], List[Tuple[int, str, float]]]] = []
    Z: List[Tuple[Tuple[List[int], List[Tuple[int, str, float]]], List[int]]] = []
    W: List[List[float]] = []
    PSI: List[EventNode] = []
    e: List[float] = []
    a: List[float] = []
    X.append(([0] * n, X_0))
    Z.append((X[0], Y))
    W.append([1 / n] * n)

    while True:
        n = len(W[t])
        if n == 0:
            print("no more sequences")
            break

        # find the best weak hypothesis
        PSI.append(Best_tree(W[t], Z[t][0][0], Z[t][0][1], Z[t][1], run_parallel=True))

        # get the weak hypothesis error
        e.append(
            sum(
                [
                    W[t][i]
                    for i in range(n)
                    if predict(PSI[t], (X[t][0][i], X[t][1][i])) != Y[i]
                ]
            )
        )

        # choose the weak hypothesis weight
        if e[t] == 0:  # there could be no classification error
            a.append(math.inf)
            break
        else:
            a.append(0.5 * np.log((1 - e[t]) / e[t]))

        if t >= iterations - 1:
            break

        W.append([0.0] * n)
        # update the weights (from exercise)
        for j in range(n):
            if Y[j] * predict(PSI[t], (X[t][0][j], X[t][1][j])) >= 0:
                W[t + 1][j] = (
                    0.5
                    * W[t][j]
                    / sum(
                        [
                            W[t][i]
                            for i in range(n)
                            if predict(PSI[t], (X[t][0][j], X[t][1][j])) * Y[i] >= 0
                        ]
                    )
                )
            else:
                W[t + 1][j] = (
                    0.5
                    * W[t][j]
                    / sum(
                        [
                            W[t][i]
                            for i in range(n)
                            if predict(PSI[t], (X[t][0][j], X[t][1][j])) * Y[i] <= 0
                        ]
                    )
                )

        # update the weights (from paper)
        # updating the weights with this formula works better than the one from the exercise,
        # Noting that at increasing the number of iterations, the prediction accuracy starts to oscillate oround some value
        # In "robot" dataset, using this formula, the accuracy is around 0.8, while using the formula from the exercise, the accuracy is around 0.6
        # for j in range(n):
        #    W[t + 1][j] = W[t][j] * np.exp(
        #        -a[t] * Y[j] * predict(PSI[t], (X[t][0][j], X[t][1][j]))
        #    )
        # W[t + 1] = [w / sum(W[t + 1]) for w in W[t + 1]]

        # update dataset
        X.append(list(zip(*[consume(PSI[t], x) for x in zip(X[t][0], X[t][1])])))
        X[t + 1], W[t + 1], Y = fix_after_consume(X[t + 1], W[t + 1], Y)
        Z.append((X[t + 1], Y))
        t += 1

    return PSI, a, e


def predict_boost(PSI, a, x):
    # print(f"predict_boost with {len(PSI)} weak hypotheses")
    # prediction without consumption
    # res = 0
    # i = 0
    # pred = 0
    # while True:
    #    pred = predict(PSI[i], (0, x))
    #    res += a[i] * pred
    #    i += 1
    #    if i >= len(PSI):
    #        # print(f"weighted prediction = {res}")
    #        break
    ### print(f"weighted prediction = {res}")
    # return res

    # prediction with consumption
    tmp = 0
    i = 0
    vt = 0
    pred = 0
    while True:
        pred = predict(PSI[i], (vt, x))
        tmp += a[i] * pred
        # print(f"pred = {pred}, a = {a[i]}, res = {res}, where y = {y}, x = {x}")
        vt, x = consume(PSI[i], (vt, x))
        i += 1
        if len(x) == 0 or i >= len(PSI):
            # print(f"weighted prediction = {res}")
            break
    return tmp


def prob_boost(PSI, a, x, y):
    # probability of the sample x to be of class y
    tmp = 0
    i = 0
    vt = 0
    prob = 0
    while True:
        pred = predict(PSI[i], (vt, x))
        if pred * y > 0:
            prob += a[i]
        vt, x = consume(PSI[i], (vt, x))
        i += 1
        if len(x) == 0 or i >= len(PSI):
            break
    if prob == math.inf and sum(a[:i]) == math.inf:
        return 1
    return prob / sum(a[:i])


def main():
    df = parse_dataset(DATASET_PATH)
    train = df.sample(frac=0.8)
    test = df.drop(train.index)

    X = train["s"].tolist()
    Y = train["y"].tolist()
    t, a, _ = ada_boost(X, Y, ITERATIONS)

    # test the model
    test_X = test["s"].tolist()
    test_Y = test["y"].tolist()
    predictions = [predict_boost(t, a, x) for x, y in zip(test_X, test_Y)]
    for i, p in enumerate(predictions):
        print(f"prediction = {p}, real value = {test_Y[i]}")

    ## a prediction is correct if the sign of the prediction is the same as the sign of the real value
    accuracy = sum(
        [1 for i in range(len(test_Y)) if predictions[i] * test_Y[i] > 0]
    ) / len(test_Y)

    print(f"Accuracy = {accuracy}")

    # probability of the sample x to be of class 1
    p_plus = [prob_boost(t, a, x, 1) for x in test_X]

    print(f"p_plus = {p_plus}")
    # probability of the sample x to be of class -1
    p_minus = [prob_boost(t, a, x, -1) for x in test_X]
    print(f"p_minus = {p_minus}")


if __name__ == "__main__":
    main()
