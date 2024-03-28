import matplotlib.pyplot as plt
import numpy as np
from ada_boost import ada_boost, predict_boost
from tree import *
from util import *

ITERATIONS = 40

# DATASET_PATH = "../datasets/activity.txt"  # 35 lines
# DATASET_PATH = "../datasets/question.txt"  # 1730 lines
# DATASET_PATH = "../datasets/epitope.txt"  # 2392 lines
# DATASET_PATH = "../datasets/gene.txt"  # 2942 lines
DATASET_PATH = "../datasets/robot.txt"  # 4302 lines


def plot_iterations_accuracy(df, iterations):
    train = df.sample(frac=0.8)
    test = df.drop(train.index)

    X = train["s"].tolist()
    Y = train["y"].tolist()

    # train a boosting classifier with a given number of iterations
    t, a, e = ada_boost(X, Y, iterations)

    accuracy = []
    for i in range(1, len(t) + 1):
        test_X = test["s"].tolist()
        test_Y = test["y"].tolist()
        print(f"iteration = {i}")
        print(f"text_y = {test_Y}")
        predictions = [
            predict_boost(t[:i], a[:i], x, y) for x, y in zip(test_X, test_Y)
        ]
        print(f"predictions = {predictions}")
        accuracy.append(
            sum([1 for i in range(len(test_Y)) if predictions[i] * test_Y[i] > 0])
            / len(test_Y)
        )
        print(f"Accuracy = {accuracy[-1]}")

    print(f"accuracy = {accuracy}")
    plt.plot(range(1, len(t) + 1), accuracy, label="accuracy")
    plt.plot(range(1, len(t) + 1), a, label="alpha")
    plt.plot(range(1, len(t) + 1), e, label="error")
    # legend
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.title("AdaBoost on " + DATASET_PATH.split("/")[-1])

    plt.show()


if __name__ == "__main__":
    df = parse_dataset(DATASET_PATH)
    plot_iterations_accuracy(df, ITERATIONS)
