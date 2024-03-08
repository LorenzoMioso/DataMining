from math import log2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tree import *
from util import *

DATASET_PATH = "../datasets/activity.txt"  # 35 lines
# DATASET_PATH = "../datasets/question.txt"  # 1730 lines
# DATASET_PATH = "../datasets/epitope.txt" # 2392 lines
DATASET_PATH = "../datasets/gene.txt"  # 2942 lines
# DATASET_PATH = "../datasets/robot.txt" # 4302 lines
ITERATIONS = 30


def fix_after_consume(X, W, Y):
    n = len(W)
    # print("fixing after consume")
    # remove empty sequences from X and adjust Z, W, Y
    empty_sequences = [i for i in range(n) if len(X[1][i]) == 0]
    # print(f"empty_sequences = {empty_sequences}")
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
    print(f"call to adaBoost with iterations = {iterations}")
    # train a boosting classifier
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
        # print(f"n = {n}")
        # print(f"Z[{t}] = {Z[t]}")
        # print(f"W[{t}] = {W[t]}")
        # print(f"sum(W[{t}]) = {sum(W[t])}")
        # train a weak learner
        PSI.append(Best_tree(W[t], Z[t][0][0], Z[t][0][1], Z[t][1]))
        # print_tree(PSI[t])
        # for i in range(n):
        #    print(
        #        f"(w = {W[t][i]}, y = {Y[i]}, prediction = {predict(PSI[t], (X[t][0][i], X[t][1][i]))}, vt = {X[t][0][i]}, s_x = {X[t][1][i]})"
        #    )
        # for i in range(n):
        #    print(
        #        f"predict(PSI[{t}], X[{t}][1][{i}]) = {predict(PSI[t], (X[t][0][i],  X[t][1][i]))}, with weight {round(W[t][i],3)}, prediction is correct = {predict(PSI[t],  (X[t][0][i],  X[t][1][i])) == Y[i]}"
        #    )
        # percentage of correct predictions
        # print(
        #    f"percentage of correct predictions = {sum([1 for i in range(n) if predict(PSI[t], (X[t][0][i],  X[t][1][i])) == Y[i]]) / n}"
        # )

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
        # print(f"e[{t}] = {e[t]}")
        # choose the weak hypothesis weight
        if e[t] == 0:  # there could be no classification error
            a.append(0.5)
            break
        else:
            a.append(0.5 * np.log((1 - e[t]) / e[t]))
        print(f"a[{t}] = {a[t]}")

        # print(f"################# end of iteration {t} #################")
        if t >= iterations - 1:
            break
        # print(f"t = {t}")

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
        # for j in range(n):
        #    W[t + 1][j] = W[t][j] * np.exp(
        #        -a[t] * Y[j] * predict(PSI[t], (X[t][0][j], X[t][1][j]))
        #    )
        # W[t + 1] = [w / sum(W[t + 1]) for w in W[t + 1]]

        # print(f"W[{t + 1}] = {W[t + 1]}")

        # update dataset
        X.append(list(zip(*[consume(PSI[t], x) for x in zip(X[t][0], X[t][1])])))
        # TODO: fix X, Z, W, Y after consume
        X[t + 1], W[t + 1], Y = fix_after_consume(X[t + 1], W[t + 1], Y)
        Z.append((X[t + 1], Y))
        t += 1

    print("################# end of training #################")
    # print("computed values")
    # for i in range(t):
    #    # print(f"PSI[{i}] = {PSI[i]}")
    #    # print(f"a[{i}] = {a[i]}")
    #    print(f"e[{i}] = {e[i]}")
    return PSI, a


def predict_boost(PSI, a, x, y):
    # print(f"call to predictBoost with PSI = {PSI}, a = {a}, x = {x}, y = {y}")
    # return y * sum([a[i] * predict(PSI[i], x) for i in range(len(PSI))]) / sum(a)
    # return sum([a[i] * predict(PSI[i], x) for i in range(len(PSI))])

    # prediction without consumption
    res = 0
    i = 0
    pred = 0
    while True:
        pred = predict(PSI[i], (0, x))
        res += a[i] * pred
        # print(f"pred = {pred}, a = {a[i]}, res = {res}, where y = {y}, x = {x}")
        i += 1
        if i >= len(PSI):
            # print(f"weighted prediction = {res}")
            break
    return res

    # prediction with consumption
    # res = 0
    # i = 0
    # vt = 0
    # pred = 0
    # while True:
    #    pred = predict(PSI[i], (vt, x))
    #    res += a[i] * pred
    #    # print(f"pred = {pred}, a = {a[i]}, res = {res}, where y = {y}, x = {x}")
    #    vt, x = consume(PSI[i], (vt, x))
    #    i += 1
    #    if len(x) == 0 or i >= len(PSI):
    #        # print(f"weighted prediction = {res}")
    #        break
    # return res


def plot_iterations_accuracy(df, iterations):
    train = df.sample(frac=0.8)
    test = df.drop(train.index)

    W = np.ones(len(train)) / len(train)
    VT = np.zeros(len(train))
    X = train["s"].tolist()
    Y = train["y"].tolist()

    accuracy = []
    for i in range(1, iterations + 1):
        t, a = ada_boost(X, Y, i)
        predictions = [
            predict_boost(t, a, x, y)
            for x, y in zip(test["s"].tolist(), test["y"].tolist())
        ]
        accuracy.append(
            sum(
                [
                    1
                    for i in range(len(test["y"].tolist()))
                    if predictions[i] * test["y"].tolist()[i] > 0
                ]
            )
            / len(test["y"].tolist())
        )
        print(f"Accuracy after {i} iterations = {accuracy[-1]}")

    plt.plot(range(1, iterations + 1), accuracy)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.show()


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

    # t, a = ada_boost(X, Y, ITERATIONS)

    # test the model
    # test_X = test["s"].tolist()
    # test_Y = test["y"].tolist()
    # predictions = [predict_boost(t, a, x, y) for x, y in zip(test_X, test_Y)]
    # for i, p in enumerate(predictions):
    #    print(f"prediction = {p}, real value = {test_Y[i]}")

    ## a prediction is correct if the sign of the prediction is the same as the sign of the real value
    # accuracy = sum(
    #    [1 for i in range(len(test_Y)) if predictions[i] * test_Y[i] > 0]
    # ) / len(test_Y)

    # print(f"Accuracy = {accuracy}")

    plot_iterations_accuracy(df, ITERATIONS)


if __name__ == "__main__":
    main()


# TODO :
# - try on other datasets
# - compute differently the v in s
